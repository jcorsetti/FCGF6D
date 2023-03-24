import argparse
import os

import json
import numpy as np
from PIL import Image

'''
Create ground truth csv of current partition, to be used from compute_metrics.py script
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Absolute path of dataset path')
    parser.add_argument('--split', type=str, default='test', help='Partition for which ground truth is computed')
    
    args = parser.parse_args()
    args.path = os.readlink('data_{}'.format(args.dataset))
    return args


def main():
    args = parse_args()
    with open(os.path.join(args.path, 'models', 'models_info.json')) as file_json:
        gt_bbox = json.load(file_json)

    # Get list of objects
    obj_ids = [int(obj_id) for obj_id in gt_bbox.keys()]

    result_file = open('gt_{}_{}.csv'.format(args.split,os.path.basename(os.path.realpath(args.path))), 'w')
    result_file.write('scene_id,im_id,obj_id,score,R,t,time\n')

    for scene in os.listdir(os.path.join(args.path, args.split)):
        with open(os.path.join(args.path, args.split, scene, 'scene_gt.json')) as json_file:
            annots = json.load(json_file)
        with open(os.path.join(args.path, args.split, scene, 'scene_gt_info.json')) as json_file:
            meta = json.load(json_file)


        for img_id, obj_list in annots.items():
            for idx, obj in enumerate(obj_list):
                obj_id = obj['obj_id']
                if obj_id in obj_ids:
                    if "{:06d}.png".format(int(img_id)) in os.listdir(os.path.join(args.path, args.split, scene, 'rgb')):
                        
                        obj_meta = meta[img_id][idx]
                        if obj_meta['visib_fract'] > 0.:

                            rot = np.array(obj['cam_R_m2c'], dtype=np.float32)
                            trasl = np.array(obj['cam_t_m2c'], dtype=np.float32) 
                            result_file.write('{0:d},{1:d},{2:d},1.0,'.format(
                                int(scene), int(img_id), int(obj_id)))
                            result_file.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f},'.format(
                                rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]))
                            result_file.write('{:.6f} {:.6f} {:.6f}, {:.6f}\n'.format(
                                trasl[0], trasl[1], trasl[2], 1.0))

if __name__ == '__main__':
    main()