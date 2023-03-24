import os
import argparse
import math
import json
from os import mkdir, listdir, readlink
from os.path import join
import random
from shutil import copy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Path of the data')
    parser.add_argument('--source_split', type=str, help='Name of source partition')
    parser.add_argument('--target_split', type=str, help='Name of target partition')
    parser.add_argument('--num', type=int, default=30000, help='How many images to sample')
    parser.add_argument('--start_part', type=int, default=0, help='Index of starting subpartition')
    
    args = parser.parse_args()
    args.path = readlink('data_{}'.format(args.dataset))

    return args

def get_all_annots(root):

    all_gt, all_gt_info, all_cameras, = {}, {}, {}
    samples = set()

    for split_key in listdir(root):

        with open(join(root,split_key,'scene_gt.json')) as f:
            cur_gt = json.load(f)
        
        with open(join(root,split_key,'scene_gt_info.json')) as f:
            cur_gt_info = json.load(f)
    
        with open(join(root,split_key,'scene_camera.json')) as f:
            cur_camera = json.load(f)
        
        all_gt[split_key] = cur_gt
        all_gt_info[split_key] = cur_gt_info
        all_cameras[split_key] = cur_camera

        for img_key in cur_gt.keys():
            img_id = '{:06d}'.format(int(img_key))
            samples.add((split_key,img_id))
    
    return all_gt, all_gt_info, all_cameras, list(samples)

def probe_extension(root):

    first_part = listdir(root)[0]
    first_img = listdir(join(root,first_part,'rgb'))[0]
    _, ext = os.path.splitext(first_img)

    return ext

def main(args):

    part_nums = math.ceil(float(args.num)/1000.)
    all_gt, all_gt_info, all_cameras, samples = get_all_annots(join(args.path, args.source_split))
    
    source_ext = probe_extension(join(args.path, args.source_split))
    print('Partition {} extension: {}'.format(args.source_split, source_ext))
    print('Loaded {} samples'.format(len(samples)))

    if args.target_split in listdir(args.path):
        print(f'Split {args.target_split} exists, adding more partitions.')
    else:
        print(f'Creating split {args.target_split}')
        mkdir(join(args.path, args.target_split))

    for part_id in tqdm(range(part_nums)):

        target_part = '{:06d}'.format(part_id+args.start_part)
        part_gt, part_gt_info, part_camera = {},{},{}

        target_root = join(args.path, args.target_split, target_part)

        mkdir(target_root)
        mkdir(join(target_root, 'depth_hf'))
        mkdir(join(target_root, 'mask_segm'))
        mkdir(join(target_root, 'rgb'))

        for img_id in range(1000):

            target_img = '{:06d}'.format(img_id)

            source_id = random.randrange(0,len(samples)-1)
            src_part, src_img =  samples.pop(source_id)

            source_root = join(args.path, args.source_split, src_part)
            
            # set ground truths
            part_gt[str(int(target_img))] = all_gt[src_part][str(int(src_img))]
            part_gt_info[str(int(target_img))] = all_gt_info[src_part][str(int(src_img))]
            part_camera[str(int(target_img))] = all_cameras[src_part][str(int(src_img))]
            
            copy(join(source_root, 'rgb', src_img+source_ext), join(target_root, 'rgb', target_img+'.png'))
            copy(join(source_root, 'mask_segm', src_img+'.png'), join(target_root, 'mask_segm', target_img+'.png'))
            copy(join(source_root, 'depth_hf', src_img+'.png'), join(target_root, 'depth_hf', target_img+'.png'))

        with open(join(target_root,'scene_gt.json'),'w') as f:
            json.dump(part_gt,f)
        
        with open(join(target_root,'scene_gt_info.json'),'w') as f:
            json.dump(part_gt_info,f)
        
        with open(join(target_root,'scene_camera.json'),'w') as f:
            json.dump(part_camera,f)





    

if __name__ == '__main__':
    args = parse_args()
    main(args)
