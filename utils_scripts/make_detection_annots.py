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

    
    for part in listdir(join(args.path,args.split)):

        subroot = join(args.path, args.split, part)

        with open(join(subroot, 'scene_gt_info.json')) as f:
            gt_info = json.load(f)

        with open(join(subroot, 'scene_gt.json')) as f:
            gt = json.load(f)

        detections = {}
        
        for img_id in gt.keys():

            gt_image = gt[img_id]
            gt_info_image = gt_info[img_id]
            detections[int(img_id)] = []
            
            for gt_obj, gt_info_obj in zip(gt_image, gt_info_image):
                obj_id = gt_obj['obj_id']
                bbox_det = gt_info_obj['bbox_visib']
                detections[int(img_id)].append({
                    'obj_id': int(obj_id),
                    'box': bbox_det
                })
    
        with open(join(subroot,'gt_detection.json'),'w') as f:
            json.dump(detections, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)


