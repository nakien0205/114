"""Still-image false-box diagnostics; threshold selection uses validation only."""
import numpy as np


def ious(box, targets):
    lo = np.maximum(box[:2], targets[:,:2])
    hi = np.minimum(box[2:], targets[:,2:])
    inter = np.maximum(hi-lo,0).prod(1)
    union = np.maximum(box[2:]-box[:2],0).prod() + np.maximum(targets[:,2:]-targets[:,:2],0).prod(1) - inter
    return inter / np.maximum(union,1e-12)


def evaluate(rows):
    events = []
    tiny_gt = small_gt = 0
    negatives = set()
    for row in rows:
        ids = [i for i,c in enumerate(row['target_cls']) if c == 0]
        gt = np.array([row['gt_boxes'][i] for i in ids]).reshape(-1,4)
        sizes = np.array([row['gt_size640'][i] for i in ids])
        tiny_gt += int((sizes < 16).sum())
        small_gt += int((sizes < 32).sum())
        if not ids:
            negatives.add(row['image'])
        used = np.zeros(len(ids),dtype=bool)
        order = sorted((i for i,c in enumerate(row['pred_cls']) if c == 0),key=lambda i:-row['conf'][i])
        for i in order:
            overlap = ious(np.array(row['pred_boxes'][i]),gt)
            overlap[used] = -1
            j = int(overlap.argmax()) if len(overlap) else -1
            tp = j >= 0 and overlap[j] >= .5
            if tp:
                used[j] = True
            events.append((row['conf'][i],int(tp and sizes[j]<16),int(tp and sizes[j]<32),int(not tp),row['image']))
    events.sort(key=lambda e:-e[0])
    tiny = small = fp = index = 0
    alarm = set()
    def point(conf):
        return dict(confidence=conf, tiny_fire_TP=tiny, small_fire_TP=small,
            tiny_fire_recall=tiny/tiny_gt if tiny_gt else None,
            small_fire_recall=small/small_gt if small_gt else None,
            false_fire_boxes=fp, false_fire_boxes_per100=100*fp/len(rows),
            fire_free_alarm_images=len(alarm), fire_free_images=len(negatives))
    best = point(1.000001)
    at90 = None
    while index < len(events):
        confidence = events[index][0]
        while index < len(events) and events[index][0] == confidence:
            _,t,s,f,name = events[index]
            tiny += t; small += s; fp += f
            if name in negatives:
                alarm.add(name)
            index += 1
        p = point(confidence)
        if fp <= .3 * len(rows) and (tiny,small,-fp) > (best['tiny_fire_TP'],best['small_fire_TP'],-best['false_fire_boxes']):
            best = p
        if tiny_gt and at90 is None and tiny/tiny_gt >= .9:
            at90 = p
    return dict(images=len(rows), tiny_fire_GT=tiny_gt, small_fire_GT=small_gt,
        at_30_false_boxes_per100_ceiling=best, at_90percent_tiny_recall=at90,
        rule='Greedy score-ordered IoU>=0.5 against ALL fire GT; include confidence ties; ceiling maximizes tiny TP, then small TP, then minimizes FP',
        limitation='Validation-selected descriptive thresholds; still-image boxes/alarms, not video incidents or frozen deployment performance')
