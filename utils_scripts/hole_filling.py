import os
import argparse
import cv2
import numpy as np
import png
import hf_utils
from os import mkdir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lmo', help='Path of the data')
    parser.add_argument('--split', type=str, help='Name of partition')
    
    args = parser.parse_args()
    args.path = os.readlink('data_{}'.format(args.dataset))

    return args

def hole_fill_folder(source_folder, dest_folder):
    
    # Fast fill with Gaussian blur @90Hz (paper result)
    fill_type = 'fast'
    extrapolate = True
    blur_type = 'gaussian'

    # Fast Fill with bilateral blur, no extrapolation @87Hz (recommended)
    # fill_type = 'fast'
    # extrapolate = False
    # blur_type = 'bilateral'

    # Multi-scale dilations with extra noise removal, no extrapolation @ 30Hz
    # fill_type = 'multiscale'
    # extrapolate = False
    # blur_type = 'bilateral'

    # Save output to disk or show process
    save_output = True

    if save_output:
        # Save to Disk
        show_process = False
        save_depth_maps = True
    
    
    images = os.listdir(source_folder)
    
    for i,img_path in enumerate(images):

        depth_image_path = os.path.join(source_folder, img_path)

        # Load depth projections from uint16 image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        projected_depths = np.float32(depth_image / 256.0)

        if fill_type == 'fast':
            final_depths = hf_utils.fill_in_fast(
                projected_depths, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depths, process_dict = hf_utils.fill_in_multiscale(
                projected_depths, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))


        # Save depth images to disk
        if save_depth_maps:
            depth_image_file_name = os.path.split(depth_image_path)[1]

            # Save depth map to a uint16 png (same format as disparity maps)
            file_path = dest_folder + '/' + depth_image_file_name
            
            with open(file_path, 'wb') as f:
                depth_image = (final_depths * 256).astype(np.uint16)

                # pypng is used because cv2 cannot save uint16 format images
                writer = png.Writer(width=depth_image.shape[1],
                                    height=depth_image.shape[0],
                                    bitdepth=16,
                                    greyscale=True)
                writer.write(f, depth_image)


def main():

    args = parse_args()

    root_folder = os.path.join(args.path, args.split)


    for partition in os.listdir(root_folder):

        source = os.path.join(root_folder, partition, 'depth')
        dest = os.path.join(root_folder, partition, 'depth_hf')
        mkdir(dest)
        hole_fill_folder(source, dest)


if __name__ == '__main__':
    main()
