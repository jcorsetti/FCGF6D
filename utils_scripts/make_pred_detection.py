import json
import argparse
from os import readlink, listdir
from os.path import join

SRC = 'best_predictions.json'
DEST = join('..',readlink('data_ycbv'), 'test_bop_bbox_pred.json')

new_preds = {}
with open(SRC) as f:
    src_preds = json.load(f)

for instance in src_preds:
    if instance['score'] > 0.3:
        part_id, img_id, cls_id = int(instance['partition_id']), int(instance['image_id']), int(instance['category_id'])
        instance_id = f'{part_id:06d}_{img_id:06d}_{cls_id+1:02d}'
        box = [int(n) for n in instance['bbox']]

        new_preds[instance_id] = box

with open(DEST,'w') as f:
    json.dump(new_preds, f)
