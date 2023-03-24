import os
import json
import yaml
import subprocess

SRC_ROOT = os.readlink('data_lm')
DEST_ROOT = os.readlink('data_lmo')
DEST_ROOT = os.path.join(DEST_ROOT, 'real')

FILE_LIST = ['train','test']
START_PART_IDX = 10

for i, class_folder in enumerate(os.listdir(SRC_ROOT)):
    part_idx = '{:06d}'.format(i+START_PART_IDX)
    dest_root = os.path.join(DEST_ROOT, part_idx)
    src_root = os.path.join(SRC_ROOT, class_folder)
    
    # make folder dir
    subprocess.call('mkdir {}'.format(dest_root),shell=True)
    # copy depth and rgb data
    subprocess.call('cp -r {}/rgb {}/'.format(src_root, dest_root),shell=True)
    subprocess.call('cp -r {}/depth {}/'.format(src_root, dest_root),shell=True)

    # also annotations are the same, and metadata is just camera data
    subprocess.call('cp {}/gt.json {}/scene_gt.json'.format(src_root, dest_root),shell=True)
    subprocess.call('cp {}/info.json {}/scene_camera.json'.format(src_root, dest_root),shell=True)

    # must generate metadata annots (assuming all objects are visible here)
    images_id = os.listdir(os.path.join(dest_root,'rgb'))
    ids_list = [str(int(os.path.splitext(image_id)[0])) for image_id in images_id]
    fake_metadata = {}

    for image_id in ids_list:
        fake_metadata[image_id] = [{
            'visib_fract': 1.
        }]

    with open(os.path.join(dest_root,'scene_gt_info.json'), 'w') as f:
        json.dump(fake_metadata,f)



