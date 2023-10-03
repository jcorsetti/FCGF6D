from os.path import join
from os import readlink, listdir
import subprocess
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Path of the data')
    parser.add_argument('--split', type=str, help='Name of partition')
    
    args = parser.parse_args()
    args.path = readlink('data_{}'.format(args.dataset))

    return args

def main(args):

    with open(join(args.path, 'models', 'models_info.json')) as f:
        obj_ids = [int(obj_id) for obj_id in json.load(f).keys()]

    for split in tqdm(listdir(join(args.path, args.split))):

        split_root = join(args.path, args.split, split)
        subprocess.call('mkdir {}'.format(join(split_root,'mask_segm')), shell=True)

        with open(join(split_root, 'scene_gt.json')) as f:
            gt = json.load(f)

        with open(join(split_root, 'scene_gt_info.json')) as f:
            meta = json.load(f)

        for img_id in gt.keys():

            img_gt = gt[img_id]
            img_meta = meta[img_id]

            mask = np.zeros((480,640,len(obj_ids)), dtype=np.uint8)

            for obj_idx, obj_annot in enumerate(img_gt):
                
                obj_id = int(obj_annot['obj_id'])
                obj_visib  = img_meta[obj_idx]['visib_fract']

                if obj_visib > 0.:
                    obj_mask = Image.open(join(split_root, 'mask_visib', '{:06d}_{:06d}.png'.format(int(img_id), obj_idx))).convert('L')
                    obj_mask = np.asarray(obj_mask).copy()
                    obj_mask[obj_mask == 255] = obj_id
                    mask[:, :, obj_id-1] = obj_mask
            
            # Reduce last dimension
            mask = np.max(mask, axis=2)

            # Convert from numpy array to PIL image for data augmentation
            mask = Image.fromarray(mask)
            mask.save(join(split_root,'mask_segm','{:06d}.png'.format(int(img_id))))



if __name__ == '__main__':
    args = parse_args()
    main(args)


