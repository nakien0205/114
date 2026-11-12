/* Offline viewer + manual review. Dataset labels, model drafts, and human boxes
   are separate sources; only an explicit human Save labels enters training. */
(() => {
'use strict';
const $=id=>document.getElementById(id),data=JSON.parse($('data').textContent);
const frames=data.frames,ns='http://www.w3.org/2000/svg',view=$('view'),seek=$('seek');
const frameMap=new Map(frames.map(f=>[f[0],f]));
const storageKey='fire_smoke_manual_review_v1:'+data.video_path+':'+data.source_sha256;
let selected=[],pos=0,state=emptyState(),localPending=false,ready=false,disk=false,browserStorage=true;
let draft=[],draftFrame=null,chosen=-1,dirty=false,drag=null,queue=Promise.resolve(),prelabels=false,pendingWrites=0;
function emptyState(){return {schema_version:1,clip_id:data.clip_id,video_path:data.video_path,
 source_sha256:data.source_sha256,revision:0,flags:[],annotations:{}}}
function say(text,bad=false){$('saveStatus').textContent=text;$('saveStatus').className=bad?'error':''}
function current(){return selected.length?frames[selected[pos]]:null}
function time(t){const ms=Math.round(t*1000),h=Math.floor(ms/3600000),m=Math.floor(ms/60000)%60,s=Math.floor(ms/1000)%60;
 return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms%1000).padStart(3,'0')}`}
function cache(pending){try{localStorage.setItem(storageKey,JSON.stringify({state,disk_pending:pending}));browserStorage=true;return true}
 catch(e){browserStorage=false;say('Browser storage unavailable. Use Export backup; '+(disk?'disk saving is still enabled.':'changes currently exist only in this tab.'),true);return false}}
function checkState(value){
 if(!value||value.schema_version!==1||value.clip_id!==data.clip_id||value.video_path!==data.video_path||value.source_sha256!==data.source_sha256)
  throw Error('This backup belongs to a different video or source version.');
 if(!Array.isArray(value.flags)||!value.annotations||Array.isArray(value.annotations)||typeof value.annotations!=='object')throw Error('Invalid review backup.');
 for(const f of value.flags){if(!f||typeof f.id!=='string'||!frameMap.has(f.frame_index)||!['false_positive','wrong_class','bad_box','miss','unsure'].includes(f.issue))throw Error('Invalid review flag.')}
 for(const [key,a] of Object.entries(value.annotations)){
  if(!a||String(a.frame_index)!==key||!frameMap.has(a.frame_index)||a.reviewed!==true||!Array.isArray(a.boxes))throw Error('Invalid manual annotation.');
  for(const b of a.boxes){if(!Array.isArray(b)||b.length!==5||![0,1].includes(b[0])||!b.slice(1).every(Number.isFinite)||b[3]<=0||b[4]<=0||b[1]-b[3]/2 < -1e-7||b[2]-b[4]/2 < -1e-7||b[1]+b[3]/2>1+1e-7||b[2]+b[4]/2>1+1e-7)throw Error('Invalid manual box.')}
 }
 return value;
}
function persist(){
 localPending=true;const localSaved=cache(true),snapshot=JSON.parse(JSON.stringify(state));
 if(!disk){say(localSaved?'Saved in this browser. Export a backup to keep a file copy.':'Unsaved on disk; export a backup now.',!localSaved);return Promise.resolve()}
 say('Saving to disk…');pendingWrites++;
 queue=queue.catch(()=>{}).then(async()=>{
  snapshot.revision=state.revision;
  const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(snapshot)});
  const result=await response.json();if(!response.ok)throw Error(result.error||'Disk save failed.');
  state.revision=result.revision;
  pendingWrites--;localPending=pendingWrites>0;cache(localPending);say(localPending?'Saving to disk…':'Saved to disk in auto_labeled/manual_reviews.');
 }).catch(error=>{pendingWrites--;localPending=true;cache(true);say('Disk save failed: '+error.message+' Export a backup before closing.',true)});
 return queue;
}
function option(parent,value,text){const item=document.createElement('option');item.value=value;item.textContent=text;parent.append(item)}
function element(tag,attrs={}){const e=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))e.setAttribute(key,value);return e}
function rectangle(b,color,label,manualIndex=null){
 const [cls,cx,cy,w,h]=b,x=(cx-w/2)*data.width,y=(cy-h/2)*data.height;
 const attrs={x,y,width:w*data.width,height:h*data.height,fill:manualIndex===chosen?'#71ff7130':'none',stroke:color,'stroke-width':Math.max(2,data.width/500)};
 const rect=element('rect',attrs);
 if(manualIndex!==null){rect.dataset.manualIndex=manualIndex;rect.style.cursor='move'}else rect.setAttribute('pointer-events','none');
 view.append(rect);
 const text=element('text',{x:Math.max(0,x),y:Math.max(18,y-5),fill:color,stroke:'#000','stroke-width':'.5','font-size':Math.max(16,data.width/65),'pointer-events':'none'});
 text.textContent=label;view.append(text);
 if(manualIndex===chosen&&$('editLabels').checked){
  for(const [corner,hx,hy] of [['nw',x,y],['ne',x+w*data.width,y],['sw',x,y+h*data.height],['se',x+w*data.width,y+h*data.height]]){
   const size=Math.max(7,data.width/100),handle=element('rect',{x:hx-size/2,y:hy-size/2,width:size,height:size,fill:'#fff',stroke:'#145f14','stroke-width':2});
   handle.dataset.handle=corner;handle.dataset.manualIndex=manualIndex;handle.style.cursor=corner+'-resize';view.append(handle);
  }
 }
}
function renderFlags(){
 const f=current(),list=$('flagList');list.replaceChildren();
 $('flagTotal').textContent=`${state.flags.length} flag(s) saved for this video`;
 if(!f)return;
 const flags=state.flags.filter(v=>v.frame_index===f[0]);
 if(!flags.length){list.textContent='No flags on this frame.';return}
 for(const flag of flags){const line=document.createElement('li'),text=document.createElement('span'),remove=document.createElement('button');
  text.textContent=`${flag.issue.replaceAll('_',' ')} — ${flag.box_index===null?'whole frame':'model box '+(flag.box_index+1)}${flag.notes?' — '+flag.notes:''} `;
  remove.textContent='Remove flag';remove.onclick=()=>{state.flags=state.flags.filter(v=>v.id!==flag.id);persist();renderFlags()};line.append(text,remove);list.append(line)}
}
function loadDraft(f){
 if(draftFrame===f[0])return;
 draftFrame=f[0];draft=JSON.parse(JSON.stringify(state.annotations[String(f[0])]?.boxes||[]));chosen=-1;dirty=false;
 $('flagNotes').value='';
}
function updateLabelStatus(){
 const f=current(),saved=f&&state.annotations[String(f[0])];
 $('labelStatus').textContent=dirty?'Unsaved box changes — click Save labels when the whole frame is correct.':saved?`Reviewed frame: ${saved.boxes.length} manual box(es) saved${saved.boxes.length?'':' (confirmed background)'}.`:'Not reviewed for training yet.';
 $('labelStatus').className=dirty?'error':'';
 $('manualSummary').textContent=`${Object.keys(state.annotations).length} reviewed frame(s) in this video`;
 $('boxSelect').replaceChildren();option($('boxSelect'),'-1','Choose a manual box');
 draft.forEach((b,i)=>option($('boxSelect'),String(i),`Box ${i+1}: ${b[0]===0?'fire':'smoke'}`));$('boxSelect').value=String(chosen);
 $('deleteBox').disabled=chosen<0;$('applyClass').disabled=chosen<0;
 $('downloadLabel').disabled=!saved;
 $('exportTraining').disabled=!disk||!Object.keys(state.annotations).length;
}
function render(){
 $('seekStatus').textContent='';
 view.replaceChildren();seek.max=Math.max(0,selected.length-1);seek.value=pos;
 $('previous').disabled=pos<=0;$('next').disabled=pos>=selected.length-1;
 const f=current();for(const id of ['saveFlag','saveLabels','clearLabels','copyModel'])$(id).disabled=!f||!ready;
 if(!f){$('meta').textContent='No frames match this filter.';$('flagTarget').replaceChildren();renderFlags();return}
 loadDraft(f);const boxes=f[3],imageUrl=data.images+encodeURIComponent(f[2]);
 view.setAttribute('viewBox',`0 0 ${data.width} ${data.height}`);view.style.touchAction=$('editLabels').checked?'none':'auto';
 view.append(element('image',{href:imageUrl,width:data.width,height:data.height,'pointer-events':'none'}));
 if($('boxes').checked)boxes.forEach((b,i)=>rectangle(b,b[0]===0?'#ff493d':'#00edff',`MODEL ${i+1}: ${b[0]===0?'FIRE':'SMOKE'}`));
 if(prelabels)for(const b of data.dataset_labels?.[String(f[0])]||[])rectangle(b,'#ffd400','DATASET '+(b[0]===0?'FIRE':'SMOKE'));
 if($('showManual').checked)draft.forEach((b,i)=>rectangle(b,'#63ff63',`MANUAL ${i+1}: ${b[0]===0?'FIRE':'SMOKE'}`,i));
 const kinds=[...new Set(boxes.map(b=>b[0]===0?'fire':'smoke'))].join(' + ')||'no model prediction';
 $('meta').textContent=`${pos+1} / ${selected.length} | ${time(f[1])} (${f[1].toFixed(6)} s) | source frame ${f[0]} | ${kinds} | ${boxes.length} model box(es)`;
 $('imageLink').href=imageUrl;$('labelLink').href=data.labels+encodeURIComponent(f[2].replace(/\.[^.]+$/,'.txt'));
 const priorTarget=$('flagTarget').value;$('flagTarget').replaceChildren();option($('flagTarget'),'frame','Whole frame');
 boxes.forEach((b,i)=>option($('flagTarget'),String(i),`Model box ${i+1}: ${b[0]===0?'fire':'smoke'}`));
 if([...$('flagTarget').options].some(o=>o.value===priorTarget))$('flagTarget').value=priorTarget;
 renderFlags();updateLabelStatus();
}
function allowNavigation(){if(!dirty)return true;return window.confirm('This frame has unsaved box changes. Leave without saving these changes?')}
function goPosition(value){if(!allowNavigation())return;pos=Math.max(0,Math.min(selected.length-1,value));draftFrame=null;render()}
document.addEventListener('keydown',event=>{
 if(event.defaultPrevented||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||drag||!ready||!selected.length)return;
 if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;
 const focus=event.target;
 if(focus?.isContentEditable||focus?.closest?.('input,textarea,select,[contenteditable="true"],[role="textbox"]'))return;
 event.preventDefault();goPosition(pos+(event.key==='ArrowRight'?1:-1));
});
function filter(){if(!allowNavigation())return;
 const cur=current()?.[1]||0,mode=$('filter').value;
 selected=frames.map((f,i)=>[f,i]).filter(([f])=>mode==='all'||(mode==='flags'?state.flags.some(v=>v.frame_index===f[0]):mode==='reviewed'?Boolean(state.annotations[String(f[0])]):mode==='none'?f[3].length===0:mode==='positive'?f[3].length>0:f[3].some(b=>b[0]===Number(mode)))).map(p=>p[1]);
 const found=selected.findIndex(i=>frames[i][1]>=cur);pos=found<0?Math.max(0,selected.length-1):found;draftFrame=null;render();
}
function point(event){const p=view.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform(view.getScreenCTM().inverse());
 return [Math.max(0,Math.min(1,q.x/data.width)),Math.max(0,Math.min(1,q.y/data.height))]}
function fromCorners(cls,x1,y1,x2,y2){const left=Math.min(x1,x2),top=Math.min(y1,y2);return [cls,(left+Math.max(x1,x2))/2,(top+Math.max(y1,y2))/2,Math.abs(x2-x1),Math.abs(y2-y1)]}
view.addEventListener('pointerdown',event=>{
 if(!$('editLabels').checked||!current()||!ready||event.button!==0)return;
 $('showManual').checked=true;event.preventDefault();const p=point(event),target=event.target,idx=target.dataset.manualIndex;
 if(target.dataset.handle){chosen=Number(idx);drag={type:'resize',index:chosen,corner:target.dataset.handle,start:p,old:[...draft[chosen]]}}
 else if($('drawMode').value==='select'&&idx!==undefined){chosen=Number(idx);drag={type:'move',index:chosen,start:p,old:[...draft[chosen]]}}
 else if($('drawMode').value==='draw'){chosen=draft.length;draft.push([Number($('manualClass').value),p[0],p[1],0,0]);drag={type:'new',index:chosen,start:p}}
 else {chosen=-1;render();return}
 view.setPointerCapture(event.pointerId);render();
});
view.addEventListener('pointermove',event=>{
 if(!drag)return;const p=point(event);
 if(drag.type==='new')draft[drag.index]=fromCorners(Number($('manualClass').value),...drag.start,...p);
 else if(drag.type==='move'){const b=drag.old;draft[drag.index]=[b[0],Math.max(b[3]/2,Math.min(1-b[3]/2,b[1]+p[0]-drag.start[0])),Math.max(b[4]/2,Math.min(1-b[4]/2,b[2]+p[1]-drag.start[1])),b[3],b[4]]}
 else {const b=drag.old,x1=b[1]-b[3]/2,x2=b[1]+b[3]/2,y1=b[2]-b[4]/2,y2=b[2]+b[4]/2;
  draft[drag.index]=fromCorners(b[0],drag.corner.includes('w')?p[0]:x1,drag.corner.includes('n')?p[1]:y1,drag.corner.includes('e')?p[0]:x2,drag.corner.includes('s')?p[1]:y2)}
 dirty=true;render();
});
function endDrag(event){if(!drag)return;const b=draft[drag.index];
 if(b[3]*data.width<2||b[4]*data.height<2){if(drag.type==='new')draft.splice(drag.index,1);else draft[drag.index]=drag.old;chosen=-1}
 drag=null;dirty=true;if(view.hasPointerCapture(event.pointerId))view.releasePointerCapture(event.pointerId);render();}
view.addEventListener('pointerup',endDrag);view.addEventListener('pointercancel',endDrag);
$('filter').onchange=filter;$('boxes').onchange=render;$('showManual').onchange=render;
$('editLabels').onchange=render;$('boxSelect').onchange=()=>{chosen=Number($('boxSelect').value);render()};
$('prelabel').onclick=()=>{prelabels=!prelabels;$('prelabel').setAttribute('aria-pressed',String(prelabels));$('prelabelStatus').hidden=!prelabels;render()};
$('previous').onclick=()=>goPosition(pos-1);$('next').onclick=()=>goPosition(pos+1);seek.oninput=()=>goPosition(Number(seek.value));
$('go').onclick=()=>{const raw=$('jump').value,target=Number(raw);
 if(raw.trim()===''||!Number.isFinite(target)||target<0){$('seekStatus').textContent='Enter a time of zero seconds or later.';return}
 const found=selected.findIndex(i=>frames[i][1]>=target);
 if(found<0){$('seekStatus').textContent='No frame at or after that time in this filter. Choose All sampled frames or an earlier time.';return}
 goPosition(found);$('seekStatus').textContent=`Showing ${current()[1].toFixed(3)} s; keyframes are sampled.`;
};
$('saveFlag').onclick=()=>{const f=current();if(!f)return;
 const target=$('flagTarget').value,issue=$('flagIssue').value,box_index=target==='frame'?null:Number(target);
 const id=`${f[0]}:${target}:${issue}`,flag={id,frame_index:f[0],timestamp_sec:f[1],box_index,issue,notes:$('flagNotes').value.trim(),updated_utc:new Date().toISOString()};
 const i=state.flags.findIndex(v=>v.id===id);if(i<0)state.flags.push(flag);else state.flags[i]=flag;persist();renderFlags();
};
$('copyModel').onclick=()=>{if(!current())return;draft=JSON.parse(JSON.stringify(current()[3]));chosen=-1;dirty=true;$('showManual').checked=true;render()};
$('clearLabels').onclick=()=>{draft=[];chosen=-1;dirty=true;render()};
$('deleteBox').onclick=()=>{if(chosen<0)return;draft.splice(chosen,1);chosen=-1;dirty=true;render()};
$('applyClass').onclick=()=>{if(chosen<0)return;draft[chosen][0]=Number($('manualClass').value);dirty=true;render()};
$('saveLabels').onclick=()=>{const f=current();if(!f)return;
 state.annotations[String(f[0])]={frame_index:f[0],timestamp_sec:f[1],boxes:JSON.parse(JSON.stringify(draft)),reviewed:true,reviewed_utc:new Date().toISOString()};
 dirty=false;persist();render();
};
$('removeReview').onclick=()=>{const f=current();if(!f)return;delete state.annotations[String(f[0])];draft=[];chosen=-1;dirty=false;persist();render()};
function download(name,body,type){const blob=new Blob([body],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('exportBackup').onclick=()=>download(data.clip_id+'_manual_review.json',JSON.stringify(state,null,2),'application/json');
$('exportFlags').onclick=()=>{const fields=['video_path','source_sha256','frame_index','timestamp_sec','box_index','issue','notes','updated_utc'];
 const cell=v=>{let s=String(v??'');if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};
 download(data.clip_id+'_review_flags.csv','\ufeff'+[fields.join(','),...state.flags.map(f=>fields.map(k=>cell(k==='video_path'?data.video_path:k==='source_sha256'?data.source_sha256:f[k])).join(','))].join('\r\n')+'\r\n','text/csv;charset=utf-8');};
$('downloadLabel').onclick=()=>{const f=current(),a=f&&state.annotations[String(f[0])];if(!a)return;
 download(f[2].replace(/\.[^.]+$/,'.txt'),a.boxes.map(b=>b[0]+' '+b.slice(1).map(v=>v.toFixed(8)).join(' ')).join('\n')+(a.boxes.length?'\n':''),'text/plain');};
$('importBackup').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;
 const imported=checkState(JSON.parse(await file.text()));
 const mergedFlags=new Map(state.flags.map(f=>[f.id,f]));for(const flag of imported.flags)mergedFlags.set(flag.id,flag);
 state.flags=[...mergedFlags.values()];Object.assign(state.annotations,imported.annotations);
 draftFrame=null;await persist();render();
 }catch(error){say('Import failed: '+error.message,true)}finally{event.target.value=''}};
$('exportTraining').onclick=async()=>{await queue;if(localPending){say('Save to disk before exporting training images.',true);return}
 location.href='/api/export-training?clip='+encodeURIComponent(data.clip_id);};
window.addEventListener('beforeunload',event=>{if(dirty||(localPending&&(disk||!browserStorage))){event.preventDefault();event.returnValue=''}});
async function init(){
 let cached=null;try{const raw=localStorage.getItem(storageKey);if(raw){cached=JSON.parse(raw);state=checkState(cached.state)}}catch(error){cached=null;state=emptyState();say('Cannot restore browser backup: '+error.message,true)}
 if(location.protocol==='http:'||location.protocol==='https:'){
  try{const response=await fetch('/api/review?clip='+encodeURIComponent(data.clip_id),{cache:'no-store'});if(!response.ok)throw Error('Local review API unavailable');
   const remote=checkState(await response.json());disk=true;
   if(cached?.disk_pending){localPending=true;say('Recovered a pending browser save; saving to disk.');await persist()}
   else {state=remote;cache(false);say('Loaded disk review. Flags and saved manual labels persist in auto_labeled/manual_reviews.')}
  }catch(error){disk=false;say('Offline mode: saves use this browser only. Start the local review launcher for disk saves and training ZIP export.',true)}
 }else say('Offline mode: saves use this browser only. Start the local review launcher for disk saves and training ZIP export.');
 ready=true;filter();
}
init();
})();
