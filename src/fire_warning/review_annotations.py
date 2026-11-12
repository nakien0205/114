"""Build timestamp CSVs and an offline box viewer from existing draft labels.

No detector rerun, source-video changes, worksheet edits, or ground-truth claims.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


CLASSES = ('fire', 'smoke')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def firesense_partition(row):
    if row.get('subset','').lower()!='firesense':return None
    parts=row['video_path'].replace('\\','/').split('/')
    if len(parts)<3 or parts[-3] not in ('fire','smoke') or parts[-2] not in ('pos','neg'):return None
    return parts[-3],parts[-2]


def parse_labels(path):
    boxes = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            values = [float(value) for value in parts[1:]]
            valid = len(values) == 4 and cls in (0, 1) and all(
                math.isfinite(value) and 0 <= value <= 1 for value in values)
            valid = valid and values[2] > 0 and values[3] > 0
        except (ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError(f'Invalid YOLO box: {path}:{number}')
        boxes.append([cls, *values])
    return boxes


def clock_time(seconds):
    millis = round(float(seconds) * 1000)
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{millis:03}'


def script_json(data):
    # File names must never be able to terminate the embedded JSON script.
    return json.dumps(data, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')


def relative_url(target, directory):
    try:
        return quote(os.path.relpath(target, directory).replace('\\', '/'), safe='/')
    except ValueError:  # Different Windows drives cannot share a relative path.
        return Path(target).resolve().as_uri()


STYLE = '''body{font:16px system-ui;margin:24px auto;padding:0 22px;max-width:1200px;background:#f4f6f8;color:#17212b}
a{color:#07599e}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}
button,input,select{font:inherit;padding:7px;margin:4px}button{cursor:pointer}small{color:#465462}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px}
svg{display:block;width:100%;max-height:70vh;background:#161d25}#seek{flex:1;min-width:200px}.notice{padding:12px;background:#fff2d3}#meta{font-weight:600;margin:12px 0}
fieldset{background:white;border:1px solid #c7d2de;border-radius:6px;margin:16px 0;padding:12px}legend{font-weight:650}textarea{font:inherit;width:min(95%,650px);padding:8px}
.error{color:#a51b14;font-weight:600}button[aria-pressed=true]{background:#fff0a3}button:disabled{cursor:default;opacity:.6}h1{overflow-wrap:anywhere}'''


VIEWER_SCRIPT = Path(__file__).with_name('review_ui.js').read_text(encoding='utf-8')


def viewer_page(row, frames, folder, index_page):
    title = f"{row.get('subset', '')}: {row.get('source_relative_path') or Path(row['video_path']).name}"
    width,height=int(row.get('width') or 0),int(row.get('height') or 0)
    if width<=0 or height<=0:
        # Older full-frame annotation worksheets did not record dimensions.
        import cv2
        image=cv2.imread(str(folder/'images'/frames[0][2])) if frames else None
        if image is None: raise ValueError(f'No usable image dimensions in {folder}')
        height,width=image.shape[:2]
    data = dict(frames=frames, width=width, height=height,
                clip_id=row.get('label_id',folder.name),video_path=row['video_path'],source_sha256=row.get('source_sha256',''),
                dataset_labels={},dataset_label_status='No dataset bounding-box label files were supplied in these local video downloads.',
                images=relative_url(folder/'images',folder.parent/'review')+'/',
                labels=relative_url(folder/'labels',folder.parent/'review')+'/')
    category=row.get('source_relative_path','').replace('\\','/').split('/')
    if row.get('subset')=='firesense' and len(category)>=3:
        data['dataset_label_status']+=f' Folder category: {category[0]}/{category[1]} (clip-level category only).'
    if row.get('dataset_labels_dir'):
        label_folder=Path(row['dataset_labels_dir']).resolve()
        count=0
        for f in frames:
            label_file=label_folder/(Path(f[2]).stem+'.txt')
            if label_file.exists():
                data['dataset_labels'][str(f[0])]=parse_labels(label_file);count+=1
        data['dataset_label_status']=f'Dataset YOLO labels explicitly supplied in {label_folder}: {count} mapped frames. Missing files remain unknown.'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{STYLE}</style><h1>{html.escape(title)}</h1>
<p><a href="{html.escape(relative_url(index_page,folder.parent/'review'),quote=True)}">All videos</a> · <a href="{html.escape(Path(row['video_path']).resolve().as_uri(), quote=True)}">Open original video</a></p>
<p class="notice">These are model predictions, not verified labels. Review the original video first when marking event times.
Use <b>All sampled frames</b> to look for misses. Unsampled source frames are not shown.
Empty labels mean no saved prediction, not verified absence. Boxes are optional to review.</p>
<div class="controls"><label>Show <select id="filter"><option value="positive">Frames with predictions</option><option value="0">Fire predictions</option><option value="1">Smoke predictions</option><option value="all">All sampled frames</option><option value="none">Frames without predictions</option><option value="flags">Flagged frames</option><option value="reviewed">Manually reviewed frames</option></select></label>
<label><input id="boxes" type="checkbox" checked>Draw boxes</label><button id="prelabel" aria-pressed="false">Pre-label</button>
<label><input id="showManual" type="checkbox" checked>Show manual labels</label></div>
<p id="prelabelStatus" hidden>{html.escape(data['dataset_label_status'])} Model drafts are shown by Draw boxes; these are not dataset ground truth.</p>
<div class="controls"><button id="previous">Previous</button><input id="seek" type="range" min="0" value="0"><button id="next">Next</button>
<label>Seek seconds <input id="jump" type="number" min="0" step="0.001" value="0" style="width:110px"></label><button id="go" title="Jump to the first displayed keyframe at or after Seek seconds">Go</button></div>
<p><small>Left/Right arrows move through displayed keyframes; typing fields keep their normal arrow keys.
Go jumps to the first frame at or after the entered time within the current Show filter. Choose All sampled frames to seek beyond detections.</small> <span id="seekStatus" role="status"></span></p>
<fieldset><legend>Manual fire/smoke labels for training</legend>
<div class="controls"><label><input id="editLabels" type="checkbox">Enable drawing</label>
<label>Tool <select id="drawMode"><option value="draw">Draw rectangle</option><option value="select">Select / move</option></select></label>
<label>Class <select id="manualClass"><option value="0">Fire</option><option value="1">Smoke</option></select></label>
<label>Manual box <select id="boxSelect"><option value="-1">Choose a manual box</option></select></label>
<button id="applyClass">Apply class</button><button id="deleteBox">Delete box</button></div>
<p>Enable drawing, choose Fire or Smoke, then drag a rectangle on the image. Use Select / move to reposition it; drag a corner to resize.
Review the whole frame before saving. Saving with zero boxes confirms a background frame.</p>
<div class="controls"><button id="copyModel">Copy model boxes as editable drafts</button><button id="clearLabels">Clear manual boxes</button>
<button id="saveLabels">Save labels</button><button id="removeReview">Remove frame from training</button><button id="downloadLabel">Download YOLO .txt</button></div>
<p id="labelStatus" role="status"></p><p id="manualSummary"></p></fieldset>
<p id="meta"></p><svg id="view" role="img" aria-label="Source keyframe with predicted bounding boxes"></svg>
<p><a id="imageLink">Original keyframe</a> · <a id="labelLink">YOLO label</a></p>
<fieldset><legend>Flag a detection problem</legend><div class="controls">
<label>Target <select id="flagTarget"><option value="frame">Whole frame</option></select></label>
<label>Problem <select id="flagIssue"><option value="false_positive">False positive</option><option value="wrong_class">Wrong class</option><option value="bad_box">Bad box</option><option value="miss">Missed fire/smoke</option><option value="unsure">Unsure</option></select></label>
<button id="saveFlag">Save flag</button></div>
<label>Notes<br><textarea id="flagNotes" rows="2" maxlength="4000" placeholder="Describe what the model got wrong"></textarea></label>
<p id="flagTotal"></p><ul id="flagList"></ul><p>Flags do not alter raw model predictions, accepted event times, or benchmark scores.</p></fieldset>
<fieldset><legend>Save and export</legend><p id="saveStatus" role="status">Loading review…</p>
<div class="controls"><button id="exportFlags">Export flags CSV</button><button id="exportBackup">Export backup JSON</button>
<label>Import backup <input id="importBackup" type="file" accept=".json,application/json"></label><button id="exportTraining" disabled>Export training ZIP (this video)</button></div>
<p>Use the local review launcher for disk saves and image+YOLO ZIP export. Directly opened HTML uses browser storage; export a backup before changing browsers or clearing storage.
Only explicitly saved manual labels enter the training ZIP.</p></fieldset>
<p><small>Inference threshold: {html.escape(row.get('model_confidence') or 'not recorded')}. Individual box confidence is unavailable in the saved five-column YOLO files.
Times are seconds from the start of the named source video. This page does not change your worksheet.</small></p>
<script id="data" type="application/json">{script_json(data)}</script><script>{VIEWER_SCRIPT}</script></html>'''


INDEX_FIELDS = ['subset','video_name','source_relative_path','video_path','frame_index','timestamp_sec','timestamp_hhmmss',
                'timestamp_quality','fire_boxes','smoke_boxes','image_path','label_path','review_page']
SUMMARY_FIELDS = ['subset','video_name','source_relative_path','video_path','sampled_frames','prediction_frames',
                  'fire_prediction_frames','smoke_prediction_frames','first_fire_sec','last_fire_sec','first_smoke_sec',
                  'last_smoke_sec','auto_confirmed_onset_sec','review_page']


def build_review(worksheet, root, destination, reuse_pages=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    review = root/'review'; review.mkdir(parents=True, exist_ok=True)
    summaries, errors, total = [], [], 0
    with (destination/'detection_index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS); writer.writeheader()
        for row in read_rows(worksheet):
            folder = root/row['label_id']; page = review/(quote(row['label_id'], safe='-_')+'.html')
            try:
                frames, positives, fire_times, smoke_times = [], 0, [], []
                if reuse_pages and page.exists():
                    old=page.read_text(encoding='utf-8')
                    cached=json.loads(old.split('<script id="data" type="application/json">')[1].split('</script>')[0])
                    cached_frames={f[2]:f[3] for f in cached['frames']}
                else:
                    cached_frames=None
                mappings = read_rows(folder/'frame_map.csv')
                # Windows directory entries already include file size. Reuse it
                # rather than issuing 600,000 separate stat filesystem requests.
                with os.scandir(folder/'labels') as entries:
                    label_sizes={entry.name:entry.stat().st_size for entry in entries if entry.name.endswith('.txt')}
                if cached_frames is None:
                    nonempty=[name for name,size in label_sizes.items() if size]
                    def read_box_file(name):
                        return name,parse_labels(folder/'labels'/name)
                    # File opens dominate large Windows annotation folders.
                    # Bounded I/O workers keep this step independent of GPU use.
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        parsed_boxes=dict(pool.map(read_box_file,nonempty))
                for entry in mappings:
                    label_name=Path(entry['image']).stem+'.txt'
                    label = folder/'labels'/label_name
                    if label_name not in label_sizes: raise FileNotFoundError(label)
                    boxes = (cached_frames[entry['image']] if cached_frames is not None else
                             parsed_boxes[label_name] if label_sizes[label_name] else [])
                    timestamp = float(entry['timestamp_sec'])
                    frames.append([int(entry['frame_index']), timestamp, entry['image'], boxes])
                    fire = sum(b[0] == 0 for b in boxes); smoke = sum(b[0] == 1 for b in boxes)
                    if fire: fire_times.append(timestamp)
                    if smoke: smoke_times.append(timestamp)
                    if not boxes: continue
                    positives += 1
                    writer.writerow(dict(subset=row.get('subset',''), video_name=Path(row['video_path']).name,
                        source_relative_path=row.get('source_relative_path',''), video_path=row['video_path'],
                        frame_index=entry['frame_index'], timestamp_sec=entry['timestamp_sec'], timestamp_hhmmss=clock_time(timestamp),
                        timestamp_quality=entry.get('timestamp_quality',''), fire_boxes=fire, smoke_boxes=smoke,
                        image_path=str(folder/'images'/entry['image']), label_path=str(label), review_page=str(page)))
                frames.sort(key=lambda f: f[0])
                page.write_text(viewer_page(row, frames, folder, destination/'review_detections.html'), encoding='utf-8')
                summaries.append(dict(subset=row.get('subset',''), video_name=Path(row['video_path']).name,
                    source_relative_path=row.get('source_relative_path',''),video_path=row['video_path'],sampled_frames=len(frames),
                    prediction_frames=positives,fire_prediction_frames=len(fire_times),smoke_prediction_frames=len(smoke_times),
                    first_fire_sec=min(fire_times) if fire_times else '',last_fire_sec=max(fire_times) if fire_times else '',
                    first_smoke_sec=min(smoke_times) if smoke_times else '',last_smoke_sec=max(smoke_times) if smoke_times else '',
                    auto_confirmed_onset_sec=row.get('auto_t_ignition_start_sec',''),review_page=str(page)))
                total += len(frames)
                print(f'Review indexed {len(summaries)}: {row["label_id"]}: {len(frames)} frames, {positives} positive', flush=True)
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(video_path=row.get('video_path',''), error=str(error)))
    with (destination/'detection_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS);writer.writeheader();writer.writerows(summaries)
    write_index(summaries,total,destination)
    report = dict(status='completed' if not errors else 'completed_with_errors',generated_utc=datetime.now(timezone.utc).isoformat(),
                  videos=len(summaries),sampled_frames=total,prediction_frames=sum(r['prediction_frames'] for r in summaries),
                  errors=errors,worksheet_modified=False,confidence_scores_available=False,reused_existing_page_boxes=reuse_pages)
    (review/'review_report.json').write_text(json.dumps(report, indent=2),encoding='utf-8')
    return report



def write_index(summaries,total,destination):
    queue_path=(Path(summaries[0]['review_page']).parent.parent/'firesense_review_queue.json') if summaries else None
    queue=json.loads(queue_path.read_text(encoding='utf-8')) if queue_path and queue_path.exists() else None
    done={s['label_id']:s for s in queue.get('videos',[])} if queue else {}
    default_scope='firesense_positive' if queue and queue.get('status')=='completed' else 'all'
    links=[]
    for row in summaries:
        def first(value): return clock_time(value) if value != '' else 'No prediction'
        group=firesense_partition(row);clip=Path(row['review_page']).stem
        scope=('firesense_positive' if group and group[1]=='pos' else 'firesense_negative' if group and group[1]=='neg' else row['subset'])
        status=('Batch flagged (folder rule)' if done.get(clip,{}).get('predicted_boxes',0) else 'No saved predictions') if clip in done else 'Manual review'
        links.append(f'<tr data-scope="{html.escape(scope,quote=True)}">'+''.join(f'<td>{value}</td>' for value in [html.escape(row['subset']),
            f'<a href="{html.escape(relative_url(Path(row["review_page"]),destination),quote=True)}">{html.escape(row["source_relative_path"] or row["video_name"])}</a>',
            first(row['first_fire_sec']),first(row['first_smoke_sec']),str(row['prediction_frames']),str(row['sampled_frames']),status])+'</tr>')
    queue_notice=(f'<p class="notice">FireSense negatives have been processed using the owner’s folder rule. The default queue contains the {int(queue["remaining_positive_clips"])} positive FireSense clips. MmodelFire frames and labels are unchanged and remain available in Show videos. Batch flags are not individually visually verified and do not create empty training labels.</p>' if queue else '')
    (destination/'review_detections.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fire and smoke detection review</title>
<style>{STYLE}</style><h1>Fire and smoke detection review</h1><p>{len(summaries)} videos · {total:,} sampled frames · {sum(int(r['prediction_frames']) for r in summaries):,} frames with predictions</p>
<p>Select a video to see its saved frames, detection times, and fire/smoke boxes. Use the time slider, Next/Previous, or seek seconds.</p>
<p class="notice">First/last predictions do not mean continuous detection. These are sampled detector outputs, not verified event times or spatial accuracy.
For independent event timestamps, watch the original video before inspecting predictions.</p>
<p><a href="detection_summary.csv">Per-video summary CSV</a> · <a href="detection_index.csv">Every positive sampled frame CSV</a> · <a href="video_stages.csv">Event worksheet</a></p>
<p>For saved flags, rectangle labeling, and image+YOLO training exports, double-click <b>Start_Detection_Review.cmd</b> in Downloads.
The local server saves manual work separately under <b>auto_labeled/manual_reviews</b>. Directly opened HTML stores work in your browser until exported.</p>
<p><a href="/api/export-training" id="exportAll" hidden>Export training ZIP — all reviewed videos</a> <span id="diskSummary"></span></p>
{queue_notice}
<p>Verify <b>t_ignition_start_sec</b> (first visible fire or smoke) and <b>should_alert_human</b> (0/1). Use -1 for reviewed absence; blank means unknown.
Leave <b>t_aftermath_start_sec</b> blank while aftermath has no independent visual definition. Its timing metrics will stay unavailable.
Keep auto_* suggestions separate. Correcting every YOLO box is optional.</p>
<label>Show videos <select id="videoScope"><option value="all">All videos</option><option value="firesense_positive">FireSense positives — manual queue</option><option value="firesense_negative">FireSense negatives — batch results</option><option value="mmodelfire">MmodelFire — unchanged</option></select></label>
<label>Find video <input id="search" type="search"></label><p id="visibleCount"></p><table><thead><tr><th>Dataset</th><th>Video</th><th>First fire</th><th>First smoke</th><th>Positive frames</th><th>Sampled frames</th><th>Review status</th></tr></thead><tbody>{''.join(links)}</tbody></table>
<script>const scope=document.getElementById('videoScope');scope.value={script_json(default_scope)};
function filterVideos(){{const q=document.getElementById('search').value.toLowerCase();let count=0;for(const r of document.querySelectorAll('tbody tr')){{r.hidden=!(scope.value==='all'||r.dataset.scope===scope.value)||!r.textContent.toLowerCase().includes(q);if(!r.hidden)count++}}document.getElementById('visibleCount').textContent=`${{count}} videos shown`;}}
document.getElementById('search').oninput=filterVideos;scope.onchange=filterVideos;filterVideos();
if(location.protocol==='http:')fetch('/api/review-status').then(r=>r.ok?r.json():Promise.reject()).then(s=>{{document.getElementById('diskSummary').textContent=`${{s.reviewed_frames}} reviewed frames and ${{s.flags}} flags saved on disk.`;document.getElementById('exportAll').hidden=!s.reviewed_frames}}).catch(()=>{{}});</script></html>''',encoding='utf-8')


def refresh_pages(worksheet,root,destination):
    root,destination=Path(root).resolve(),Path(destination).resolve()
    rows=read_rows(worksheet)
    for row in rows:
        page=root/'review'/(quote(row['label_id'],safe='-_')+'.html')
        old=page.read_text(encoding='utf-8')
        saved=json.loads(old.split('<script id="data" type="application/json">')[1].split('</script>')[0])
        page.write_text(viewer_page(row,saved['frames'],root/row['label_id'],destination/'review_detections.html'),encoding='utf-8')
    summaries=read_rows(destination/'detection_summary.csv')
    total=sum(int(r['sampled_frames']) for r in summaries)
    write_index(summaries,total,destination)
    return dict(status='completed',videos=len(rows),sampled_frames=total,mode='refresh_ui_only',worksheet_modified=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worksheet',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--refresh-pages',action='store_true',help='Refresh only HTML from existing embedded frame data; no label rescan or CSV changes')
    parser.add_argument('--reuse-pages',action='store_true',help='Reuse saved box data from existing review pages; do not use after editing labels')
    args=parser.parse_args()
    report=(refresh_pages(args.worksheet,args.root,args.destination or args.worksheet.parent) if args.refresh_pages else
            build_review(args.worksheet,args.root,args.destination or args.worksheet.parent,args.reuse_pages))
    print(json.dumps(report,indent=2))
    if report.get('errors'): raise SystemExit(1)


if __name__ == '__main__': main()
