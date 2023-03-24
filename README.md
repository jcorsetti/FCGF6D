# 6Dpose

## Setup

Create virtual environment:

```bash
conda create --name 6Dpose python=3.8
```

Install PyTorch:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Install dependencies:

```bash
pip install tensorboard open3d opencv-python pytorch_warmup
```

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
```

Do we need also scikit-image?

For FPS computation we utilize a cpp module contained in lib/ directory. It depends on the cffi package, hence it should be installed:
```bash
pip install cffi
```
To configure run the following:
```bash
cd lib/csrc/fps
python3 setup.py build_ext --inplace
```

## Preprocessing

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore run_train.py --exp 'exp??' --n_workers 2 --bs 8 --n_epochs 100 --freq_train 1 --freq_valid 10 --freq_save 10 --split_train 'train_pbr' --split_val 'test' --optim_type 'Adam' --lr 1e-3 --scheduler_type 'step' --gamma 0.1 --mu3 1.0 --seg_loss dice --use_augs norm
```

Format for augmentation list:

A string in the format aug_code1,augcode2,..,augcodeN

Where augcode is one of:

-hflip  : random horizontal flip
-vflip  : random vertical flip
-jitter : random color jittering
-norm   : normalization
-rotate : random rotation

Example:

--use_augs norm,hflip,vflip

Additional training arguments:

To train only on a single class (ex. 5):
```bash
--use_single_class 5 
```
NB: this should be used also with run_test.py script

To not use random augmentations:
```bash
--use_augs False 
```
 

## Resume train

Add --flag_resume --checkpoint 'epoch=????' to the training arguments to resume the training from a given checkpoint

## Test

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore run_test.py --exp debug_depth --n_workers 1 --bs 1 --split overfit --checkpoint 'epoch=01999'
```

By default, the above script expects a model not trained with plane mask. To evaluate a model trained with plane mask, there are two options:
```bash
--use_plane_model True // This will evaluate a plane-trained model without using plane for evalution
--use_plane_model True --use_plane_eval True // This will evaluate a plane-trained model also using plane for evalution
```

## Metrics

To compute ADD and REP metrics use:

```bash
python compute_metrics.py --exp 'exp00' --split 'test' --checkpoint 'epoch=???'
```

If not already present, ground truth should be generated before running compute_metrics:

```bash
python make_gt.py --split 'test'
```

### BOP Toolkit

In order to use the BOP Toolkit for validating metrics results, clone the repository at https://github.com/thodan/bop_toolkit and follow the instructions for installation. After that, in order to compute the wanted metrics, the file scripts/eval_bop19.py should be modified. In particular, at [line 21]( https://github.com/thodan/bop_toolkit/blob/2caac4ed3b57c78a2379d0f4f0d76e67d66718d4/scripts/eval_bop19.py#L21) are listed the errors to compute; it is sufficient to replace the default errors with the following lines
```pytrhon
p = {
  # Errors to calculate.
  'errors': [
    {
      'n_top': -1,
      'type': 'add',
      'correct_th': [[0.1]]
    },
    {
      'n_top': -1,
      'type': 'proj',
      'correct_th': [[5]]
    },
  ],
```
or with any implemented errors we want the script to compute; such errors can be found in scripts/bop_toolkit_lib/pose_error.py.
Then execute the script as reported in the repo

```bash
python3 scripts/eval_bop19.py --renderer_type=python --result_filenames=NAME_OF_CSV_WITH_RESULTS
```
Note that the name of the file containing the results should be changed; in the bop toolkit files of results should follow the convention name-of-method_datasets-split.csv, e.g. our-method_lmo-test.csv
If we want to use the toolkit to test the method on the whole test set rather than on the subset of the BOP challenge, run write_json.py:
```bash
python3 write_json.py --path PATH_TO_THE_DATASET 
```
and then proceed as explained above.

## Notes

With batch size = 16, an epoch requires 3125 iterations and takes around 2900 seconds (50 minutes).
If we do not log the gradients on tensorboard the time drops to around 2200 seconds (37 minutes).

### Struttura del dataset

LM consiste di 15 scene di test, denominate 000001, 000002, ..., 000015.
Per ogni scena viene fornita la maschera e la ground-truth solamente dell'oggetto centrale.
LM-O è stato costruito selezionando la scena 000002 ed annotando invece più oggetti (che sono un sottoinsieme di quelli di LM).

Una cosa importante da tenere presente è la convenzione utilizzato per denominare le maschere.
Per ogni immagine di test, e.g. 000003.png, ci sono tante maschere quanti sono gli oggetti presenti nell'immagine,
i.e. 000003_000001.png, 000003_000002.png, ..., 000003_000007.png.
Dal momento che non per forza tutti gli oggetti sono presenti nell'immagine,
il numero dopo l'underscore indica solamente l'ordine relativo degli oggetti nella lista degli oggetti del dataset
e per capire di che oggetto si tratta dobbiamo quindi fare riferimento alle grount-truths contenute in scene_gt.json.

### Procedura di training

Abbiamo notato che sul sito di BOP vengono fornite delle immagini di training sintetiche create utilizzando una procedura analoga a quella del paper di riferimento *Segmentation-driven 6D Object Pose Estimation*.
Possono essere trovate [qui](https://thodan.github.io/objectsynth/).
