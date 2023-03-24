import json
import argparse
from os import readlink, listdir
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Absolute path of dataset path')
    parser.add_argument('--split', type=str, default='test', help='Partition for which ground truth is computed')
    
    args = parser.parse_args()
    args.path = readlink('data_{}'.format(args.dataset))
    return args


def main(args):

    detections = {}
    
    for part in listdir(join(args.path,args.split)):

        subroot = join(args.path, args.split, part)

        with open(join(subroot, 'scene_gt_info.json')) as f:
            gt_info = json.load(f)

        with open(join(subroot, 'scene_gt.json')) as f:
            gt = json.load(f)

        for img_id in gt.keys():

            gt_image = gt[img_id]
            gt_info_image = gt_info[img_id]

            for gt_obj, gt_info_obj in zip(gt_image, gt_info_image):
                obj_id = gt_obj['obj_id']
                bbox_det = gt_info_obj['bbox_visib']
                instance_id = f"{part}_{int(img_id):06d}_{obj_id:02d}"
                detections[instance_id] = bbox_det
    
    with open(join(args.path, f'{args.split}_bbox_gt.json'),'w') as f:
        json.dump(detections,f)


if __name__ == '__main__':
    args = parse_args()
    main(args)


