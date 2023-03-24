from os import listdir, readlink
from os.path import join
import subprocess
import sys

exp_data = readlink('exp_data')
exp = sys.argv[1]

subroot = join(exp_data, exp, 'runs', 'wandb')
for run in listdir(subroot):
    if run.startswith('offline-run'):
        tosync = join(subroot,run)
        subprocess.call('wandb sync ' + tosync, shell=True)