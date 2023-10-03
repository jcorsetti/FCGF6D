# 6Dpose

Welcome to the official repository of ***Revisiting Fully Convolutional Geometric Features for Object 6D Pose Estimation***, or ***FCGF6D***,
to be presented at [Recovering 6D Pose Workshop](https://cmp.felk.cvut.cz/sixd/workshop_2023/) at ICCV 2023.
The repository is still work in progress as I am simplifing and refactoring the code, but everything is here. 
I am currently busy finishing my Master's Degree, but I hope to get all the code done by end of October.
Feel free to contact me if you have any questions: jaime.corsetti98@gmail.com.

**TODOS**:
- Provide checkpoints
- Instructions to generate YCBV training set
- Add metadata (e.g. object splits)

## Setup

Our work is based on [Minkoski Engine](https://github.com/NVIDIA/MinkowskiEngine) and implemented with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
Please install them following the official instructions before proceeding.
We strongly suggest to use a virtual environment for the installation.
After this you can install all the other libraries in the requirements.txt file.

## Preprocessing

Download the YCBV test split and the OccludedLinemod (LMO) test and train_pbr split from the BOP challenge website.
All contents of YCBV should go in a data_ycbv folder, all contents of LMO should go in a data_lmo folder.

After this, generate the ground truth by running:
```bash
python utils_scripts/make_gt.py --dataset lmo --split test
python utils_scripts/make_gt.py --dataset ycbv --split test
```

and place the resulting files in the eval_gts folder.
This formats the ground truth in an easy-to-read file to be used for the compute_metrics.py script.

Generate the segmentation mask in a better format by running:
```bash
python utils_scripts/preprocessing_mask.py --dataset lmo --split test
python utils_scripts/preprocessing_mask.py --dataset ycbv --split test
```
This will turn the N binary masks for each image provided by BOP in a single mask, which speeds up the I/O processes.

Finally apply the hole filling algorithms:
```bash
python utils_scripts/hole_filling.py --dataset lmo --split test
python utils_scripts/hole_filling.py --dataset ycbv --split test
```
to generate the hole filled depth masks as in PVN3D and FFB6D.
Hole filling algorithm and mask preprocessing should be carried out also for any training splits.

## Train

For OccludedLinemod and YCBV trainings

```bash
CUDA_VISIBLE_DEVICES=0 python run_train.py --exp exp_lmo --dataset lmo --split_train 'train_pbr' --split_val 'test' --bs 8 --n_epochs 10 --freq_valid 2 --freq_save 2  --augs_erase True --augs_rgb True --depth_points 50000 --arch mink34
CUDA_VISIBLE_DEVICES=0 python run_train.py --exp exp_ycbv --dataset ycbv --split_train 'custom_train' --split_val 'test' --bs 8 --n_epochs 110 --freq_valid 10 --freq_save 10  --augs_erase True --augs_rgb True --depth_points 20000 --arch mink50
```

please see utils/parsing.py (parse_train_args function) for the complete list of options at training time.

## Test

```bash
CUDA_VISIBLE_DEVICES=0 python --exp exp_lmo --depth_points 50000 --arch mink34 --checkpoint 'epoch=0009' --split test --solver teaser
CUDA_VISIBLE_DEVICES=0 python --exp exp_ycbv --depth_points 20000 --arch mink50 --checkpoint 'epoch=0109' --split test --solver ransac
```

please see utils/parsing.py (parse_test_args function) for the complete list of options at evaluation time.

The above scripts will produce the poses, in order to compute the metrics also the following is necessary:

## Metrics


```bash
python compute_metrics.py --exp 'exp_lmo' --split 'test' --checkpoint 'epoch=0009' --solver teaser 
python compute_metrics.py --exp 'exp_ycbv' --split 'test' --checkpoint 'epoch=0109' --solver ransac 
```

## Acknowledgements

Parts of this work was based on the code provided by [PVN3D](https://github.com/ethnhe/PVN3D) (ADDS-AUC metric code) and [Fast Depth Completition](https://github.com/kujason/ip_basic) for the depth hole-filling algorithm.

## Citation

Please cite [FCGF6D](https://arxiv.org/pdf/2307.15514.pdf) if you use this repository in your publications:
```
@inproceedings{corsetti2023fcgf6d,
  author = {Corsetti, Jaime and Boscaini, Davide and Poiesi, Fabio},
  title = {Revisiting Fully Convolutional Geometric Features for Object 6D Pose Estimation},
  booktitle = {International Workshop on Recovering 6D Object Pose (R6D)},
  year = {2023}
}
```

