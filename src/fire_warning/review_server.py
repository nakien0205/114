"""Local annotation UI with separate manual reviews and YOLO training export."""
import argparse
import copy
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading
from urllib.parse import parse_qs, urlparse,unquote
import webbrowser
import zipfile

from .review_annotations import read_rows


ISSUES = {'false_positive', 'wrong_class', 'bad_box', 'miss', 'unsure'}


class RevisionConflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, worksheet, root):
        self.root = Path(root).resolve()
        self.rows = {row['label_id']: row for row in read_rows(worksheet)}
        self.directory = self.root/'manual_reviews'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.maps = {}

    def row(self, clip):
        if not isinstance(clip,str) or clip not in self.rows:
            raise ValueError('Unknown video identifier')
        return self.rows[clip]

    def mappings(self, clip):
        self.row(clip)
        if clip not in self.maps:
            self.maps[clip] = {int(r['frame_index']): r for r in read_rows(self.root/clip/'frame_map.csv')}
        return self.maps[clip]

    def load(self, clip):
        row = self.row(clip)
        path = self.directory/(clip+'.json')
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
            if state.get('source_sha256', '') != row.get('source_sha256', ''):
                raise ValueError('Source identity changed; preserve and reverify the existing manual review')
            return state
        return dict(schema_version=1, clip_id=clip, video_path=row['video_path'],
                    source_sha256=row.get('source_sha256', ''), revision=0, flags=[], annotations={})

    def frame(self, clip, value):
        if isinstance(value, bool) or not isinstance(value, int) or value not in self.mappings(clip):
            raise ValueError('Frame is not in the saved source-frame mapping')
        entry = self.mappings(clip)[value]
        return dict(frame_index=value, timestamp_sec=float(entry['timestamp_sec']))

    @staticmethod
    def boxes(value):
        if not isinstance(value, list) or len(value)>500:
            raise ValueError('Expected at most 500 boxes per frame')
        result=[]
        for box in value:
            if not isinstance(box, list) or len(box)!=5:
                raise ValueError('Each box needs class and normalized center/width/height')
            cls,*coords=box
            if isinstance(cls,bool) or cls not in (0,1) or any(
                    isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in coords):
                raise ValueError('Invalid class or non-finite box coordinates')
            cx,cy,w,h=coords
            if w<=0 or h<=0 or cx-w/2 < -1e-7 or cy-h/2 < -1e-7 or cx+w/2>1+1e-7 or cy+h/2>1+1e-7:
                raise ValueError('Boxes must have positive size and lie inside the image')
            result.append([int(cls),*map(float,coords)])
        return result

    def validate(self, clip, payload):
        row=self.row(clip)
        if not isinstance(payload,dict) or payload.get('schema_version')!=1 or payload.get('clip_id')!=clip:
            raise ValueError('Invalid review schema or video identifier')
        if payload.get('source_sha256','')!=row.get('source_sha256','') or payload.get('video_path')!=row['video_path']:
            raise ValueError('Review belongs to another source video')
        flags=payload.get('flags',[]);annotations=payload.get('annotations',{})
        if not isinstance(flags,list) or not isinstance(annotations,dict):
            raise ValueError('Invalid flags or annotations')
        clean_flags=[];ids=set()
        for flag in flags:
            if not isinstance(flag,dict) or flag.get('issue') not in ISSUES:
                raise ValueError('Invalid flag issue')
            identifier=flag.get('id')
            if not isinstance(identifier,str) or not 1<=len(identifier)<=200 or identifier in ids:
                raise ValueError('Invalid or duplicate flag identifier')
            ids.add(identifier)
            note=flag.get('notes','')
            box_index=flag.get('box_index')
            if not isinstance(note,str) or len(note)>4000:
                raise ValueError('Flag notes must be at most 4000 characters')
            if box_index is not None and (isinstance(box_index,bool) or not isinstance(box_index,int) or not 0<=box_index<500):
                raise ValueError('Invalid predicted-box index')
            clean_flags.append(dict(id=identifier,**self.frame(clip,flag.get('frame_index')),
                issue=flag['issue'],box_index=box_index,notes=note,updated_utc=str(flag.get('updated_utc',''))[:40]))
        clean_annotations={}
        for key,annotation in annotations.items():
            if not isinstance(annotation,dict) or annotation.get('reviewed') is not True:
                raise ValueError('Only explicitly reviewed frames can be saved for training')
            frame=self.frame(clip,annotation.get('frame_index'))
            if key!=str(frame['frame_index']):
                raise ValueError('Annotation key does not match its source frame')
            clean_annotations[key]=dict(**frame,reviewed=True,boxes=self.boxes(annotation.get('boxes')),
                reviewed_utc=str(annotation.get('reviewed_utc',''))[:40])
        return dict(schema_version=1,clip_id=clip,video_path=row['video_path'],source_sha256=row.get('source_sha256',''),
                    flags=clean_flags,annotations=clean_annotations)

    def save(self, clip, payload):
        with self.lock:
            current=self.load(clip)
            if payload.get('revision')!=current['revision']:
                raise RevisionConflict('This video changed in another tab. Export a backup, then reload before saving.')
            state=self.validate(clip,payload)
            state.update(revision=current['revision']+1,updated_utc=datetime.now(timezone.utc).isoformat())
            path=self.directory/(clip+'.json')
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=self.directory,suffix='.tmp',delete=False) as stream:
                temp=Path(stream.name)
                json.dump(state,stream,ensure_ascii=False,indent=2,allow_nan=False)
            try: temp.replace(path)
            finally: temp.unlink(missing_ok=True)
            return state

    def export_training(self, clips=None):
        """Only explicit manual saves enter this ZIP; raw draft labels never do."""
        clips=list(clips) if clips is not None else list(self.rows)
        records=[];states=[]
        with self.lock:
            for clip in clips:
                state=self.validate(clip,self.load(clip))
                if state['annotations']: states.append(state)
        for state in states:
            for annotation in state['annotations'].values():
                mapping=self.mappings(state['clip_id'])[annotation['frame_index']]
                source=(self.root/state['clip_id']/'images'/mapping['image']).resolve()
                allowed=(self.root/state['clip_id']/'images').resolve()
                if not source.is_relative_to(allowed) or not source.is_file():
                    raise ValueError('A reviewed source image is missing or outside its image folder')
                stem=state['clip_id']+'_'+Path(mapping['image']).stem
                records.append((state,annotation,source,stem))
        if not records: raise ValueError('No reviewed frames yet. Draw/correct boxes and click Save labels first.')
        archive=tempfile.SpooledTemporaryFile(max_size=32*1024*1024,mode='w+b')
        try:
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                manifest=[]
                for state,annotation,source,stem in records:
                    z.write(source,'images/'+stem+source.suffix)
                    text=''.join(f'{b[0]} '+' '.join(f'{v:.8f}' for v in b[1:])+'\n' for b in annotation['boxes'])
                    z.writestr('labels/'+stem+'.txt',text)
                    manifest.append(dict(video_path=state['video_path'],source_sha256=state['source_sha256'],
                        **annotation,image='images/'+stem+source.suffix,label='labels/'+stem+'.txt',
                        provenance='human_reviewed_manual_annotation'))
                z.writestr('review_manifest.json',json.dumps(manifest,indent=2))
                z.writestr('manual_reviews.json',json.dumps(states,indent=2))
                z.writestr('classes.txt','fire\nsmoke\n')
                z.writestr('data.yaml','path: .\ntrain: images\n# Set val to an independent, reviewed validation split before training.\nnames:\n  0: fire\n  1: smoke\n')
                z.writestr('README.txt','Human-reviewed YOLO training material\n\nOnly frames explicitly saved with Save labels are included.\n'
                    'An empty label is a manually reviewed background frame.\nRaw model draft boxes and issue flags are not training labels.\n'
                    'Images retain their original source resolution.\n'
                    'No validation split has been assigned. Split by independent video/physical event before training; do not use adjacent frames of the same event across train and validation.\n'
                    'Set the val entry in data.yaml after that split.\n'
                    )
            archive.seek(0)
            return archive,len(records)
        except Exception:
            archive.close();raise


class ReviewHandler(SimpleHTTPRequestHandler):
    server_version='LocalAnnotationReview/1'

    @property
    def store(self):return self.server.review_store

    def response(self, status, value):
        body=json.dumps(value,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def allowed_origin(self):
        host=self.headers.get('Host','')
        expected={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        origin=self.headers.get('Origin')
        return host in expected and (origin is None or origin in {'http://'+h for h in expected})

    def allowed_asset(self,path):
        # The document root is Downloads, but unrelated Downloads files must
        # never be exposed by this annotation service, even on localhost.
        decoded=unquote(path)
        if '\\' in decoded:return False
        parts=Path(decoded.lstrip('/')).parts
        if '..' in parts:return False
        if decoded in ('/review_detections.html','/video_stages.csv','/detection_index.csv','/detection_summary.csv'):
            return True
        if len(parts)==3 and parts[0]==self.store.root.name and parts[1]=='review' and parts[2].endswith('.html'):
            return True
        if len(parts)==4 and parts[0]==self.store.root.name and parts[1] in self.store.rows:
            return (parts[2]=='images' and parts[3].endswith('.jpg')) or (parts[2]=='labels' and parts[3].endswith('.txt'))
        return False

    def do_GET(self):
        if not self.allowed_origin():return self.response(403,dict(error='Local same-origin requests only'))
        parsed=urlparse(self.path);query=parse_qs(parsed.query)
        try:
            if parsed.path=='/api/review':
                with self.store.lock: state=copy.deepcopy(self.store.load(query.get('clip',[''])[0]))
                return self.response(200,state)
            if parsed.path=='/api/review-status':
                with self.store.lock:
                    states=[self.store.load(clip) for clip in self.store.rows]
                return self.response(200,dict(videos=len(states),reviewed_frames=sum(len(s['annotations']) for s in states),
                    flags=sum(len(s['flags']) for s in states)))
            if parsed.path=='/api/export-training':
                clips=query.get('clip')
                archive,count=self.store.export_training(clips)
                with archive:
                    archive.seek(0,2);length=archive.tell();archive.seek(0)
                    self.send_response(200);self.send_header('Content-Type','application/zip')
                    self.send_header('Content-Disposition','attachment; filename="reviewed_yolo_training.zip"')
                    self.send_header('Content-Length',str(length));self.send_header('X-Reviewed-Frames',str(count));self.end_headers()
                    shutil.copyfileobj(archive,self.wfile)
                return
            if parsed.path.startswith('/api/'):return self.response(404,dict(error='Unknown API route'))
            if parsed.path=='/favicon.ico':
                self.send_response(204);self.end_headers();return
            if not self.allowed_asset(parsed.path):return self.response(403,dict(error='Only annotation review assets are served'))
            self.send_static=True
            return super().do_GET()
        except (ValueError,OSError,KeyError) as error:return self.response(400,dict(error=str(error)))

    def end_headers(self):
        if getattr(self,'send_static',False):self.send_header('Cache-Control','no-cache')
        super().end_headers()

    def do_POST(self):
        if not self.allowed_origin():return self.response(403,dict(error='Local same-origin requests only'))
        if urlparse(self.path).path!='/api/review':return self.response(404,dict(error='Unknown API route'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=20*1024*1024:raise ValueError('Review request must be 1 byte to 20 MB')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                raise ValueError('Expected application/json')
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict):raise ValueError('Expected a JSON object')
            state=self.store.save(payload.get('clip_id'),payload)
            return self.response(200,state)
        except RevisionConflict as error:return self.response(409,dict(error=str(error)))
        except (ValueError,OSError,KeyError) as error:return self.response(400,dict(error=str(error)))


def make_server(worksheet, root, port=8766):
    root=Path(root).resolve()
    server=ThreadingHTTPServer(('127.0.0.1',port),partial(ReviewHandler,directory=str(root.parent)))
    server.review_store=ReviewStore(worksheet,root)
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worksheet',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    try:server=make_server(args.worksheet,args.root,args.port)
    except OSError as error:raise SystemExit(f'Cannot start local review server: {error}. Try another --port.')
    url=f'http://127.0.0.1:{server.server_port}/review_detections.html'
    print(f'Open {url}\nManual flags/labels: {server.review_store.directory}\nStop with Ctrl+C.',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
