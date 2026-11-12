"""Verify and stage train/validation from the frozen combined dataset."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import yaml


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def locate(search, expected, explicit=None):
    candidates = []
    if explicit:
        candidates = [Path(explicit)]
    else:
        for current, dirs, files in os.walk(search):
            folder = Path(current)
            if 'dataset-manifest.json' in files:
                candidates.append(folder)
                dirs[:] = []
            for name in files:
                if name == 'home_fire_dfire_20261102.zip':
                    candidates.append(folder / name)
            dirs[:] = [d for d in dirs if d not in ('images','labels','test','assets','vendor')]
    accepted = []
    for candidate in candidates:
        if candidate.is_dir():
            if not all((candidate / split / sub).is_dir() for split in ('train','val') for sub in ('images','labels')):
                # Companions also contain the manifest, but have no images.
                continue
            raw = (candidate / 'dataset-manifest.json').read_text()
        elif candidate.is_file() and zipfile.is_zipfile(candidate):
            with zipfile.ZipFile(candidate) as archive:
                raw = archive.read('dataset-manifest.json')
        else:
            continue
        if json.loads(raw).get('version_sha256') == expected:
            accepted.append(candidate)
    if not accepted:
        raise RuntimeError('Attach the supplied complete home_fire_dfire_20261102 dataset ZIP/folder, or set DATASET_ROOT to it.')
    return sorted(accepted, key=lambda x:(not x.is_dir(),str(x)))[0]


def prepare(root, assets, explicit=None, search='/kaggle/input'):
    root, assets = Path(root), Path(assets)
    manifest = json.loads((assets / 'dataset-manifest.json').read_text())
    entries = manifest['entries']
    actual = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert actual == manifest['version_sha256']
    source = locate(search, actual, explicit)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    staged = root / 'data'
    counts = {'train':0, 'val':0}
    try:
        for e in entries:
            if e['split'] not in counts:
                continue
            assert Path(e['image']).name == e['image']
            split = e['split']
            name = e['image']
            image_rel = f'{split}/images/{name}'
            label_rel = f'{split}/labels/{Path(name).with_suffix(".txt").name}'
            target_image, target_label = staged / image_rel, staged / label_rel
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if archive:
                with archive.open(image_rel) as incoming, target_image.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                labels = archive.read(label_rel)
            else:
                image = source / image_rel
                assert file_hash(image) == e['image_sha256'], image
                target_image.symlink_to(image.resolve())
                labels = (source / label_rel).read_bytes()
            assert hashlib.sha256(labels).hexdigest() == e['label_sha256'], label_rel
            target_label.write_bytes(labels)
            if archive:
                assert file_hash(target_image) == e['image_sha256']
            counts[split] += 1
    finally:
        if archive:
            archive.close()
    (root / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(staged.resolve()), train='train/images',
        val='val/images', nc=2, names={0:'fire',1:'smoke'}), sort_keys=False))
    record = dict(complete=True, dataset_version_sha256=actual, source=str(source), counts=counts,
        test_used=False, split_policy='Source splits preserved; no verified scene/event independence',
        cross_split_exact_duplicate_groups=len(manifest['audit']['cross_split_duplicate_groups']))
    write(root / 'data-preparation.json', record)
    return record


def capture_validator(reference):
    from ultralytics.models.yolo.detect import DetectionValidator
    class Capture(DetectionValidator):
        def run_callbacks(self, event):
            pass

        def init_metrics(self, model):
            super().init_metrics(model)
            self.records = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for index,pred in enumerate(preds):
                pb = self._prepare_batch(index,batch)
                name = Path(pb['im_file']).name
                row = copy.deepcopy(reference[name])
                assert list(pb['ori_shape']) == row['ori_shape'], name
                assert pb['cls'].int().cpu().tolist() == row['target_cls'], name
                scaled = self.scale_preds(pred,pb)
                row.update(pred_cls=pred['cls'].int().cpu().tolist(), conf=pred['conf'].cpu().tolist(),
                    pred_boxes=scaled['bboxes'].cpu().tolist())
                self.records.append(row)

        def get_stats(self):
            result = super().get_stats()
            write(self.save_dir / 'native-predictions.json', self.records)
            return result
    return Capture
