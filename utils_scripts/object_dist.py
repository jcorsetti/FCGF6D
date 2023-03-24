from os import listdir, readlink
from os.path import join
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Path of the data')
    parser.add_argument('--split', type=str, help='Name of partition')
    
    args = parser.parse_args()
    args.path = readlink('data_{}'.format(args.dataset))
    return args

def main(args):

    with open(join(args.path, 'models', 'models_info.json')) as f:
        obj_ids = [int(obj_id) for obj_id in list(json.load(f).keys())]

    obj_occ = np.zeros(len(obj_ids), dtype=int)
    obj_pixels = np.zeros(len(obj_ids))
    obj_visib = np.zeros(len(obj_ids))
    obj_valid = np.zeros(len(obj_ids))

    for split in listdir(join(args.path, args.split)):
        split_root = join(args.path, args.split, split)
        
        with open(join(split_root, 'scene_gt.json')) as f:
            gt = json.load(f)

        with open(join(split_root, 'scene_gt_info.json')) as f:
            meta_gt = json.load(f)

        for img_id, obj_list in gt.items():
            
            for idx, obj in enumerate(obj_list):

                obj_id = int(obj['obj_id'])

                if obj_id in obj_ids:
                    
                    obj_idx = obj_ids.index(obj_id)

                    if "{:06d}.jpg".format(int(img_id)) in listdir(join(split_root, 'rgb')) or "{:06d}.png".format(int(img_id)) in listdir(join(split_root, 'rgb')):
                        obj_meta = meta_gt[img_id][idx]

                        if obj_meta['visib_fract'] > 0.:
                            obj_occ[obj_idx] += 1
                        
                        obj_visib[obj_idx] += obj_meta['visib_fract']
                        obj_pixels[obj_idx] += obj_meta['px_count_visib']
                        obj_valid[obj_idx] += obj_meta['px_count_valid']
    
    total = 0
    print("Obj     Avg visib r   Avg pix   Avg visib     Num")
    for obj_idx, obj_id in enumerate(obj_ids):
        if obj_occ[obj_idx] > 0:
            avg_visib = obj_visib[obj_idx] / obj_occ[obj_idx]
            avg_pixels = obj_pixels[obj_idx] / obj_occ[obj_idx]
            avg_valid = obj_valid[obj_idx] / obj_occ[obj_idx]
            print("{:2d}      {:.3f}         {:4d}       {:4d}     {:5d}".format(int(obj_id), avg_visib, int(avg_valid), int(avg_pixels), obj_occ[obj_idx]))



if __name__ == '__main__':
    args = parse_args()
    main(args)
