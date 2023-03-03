# ML-pytorch

## Activate conda from mambaforge

The recommanded way to use mamba is [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html#)

On CBE just add following to your  ```.bashrc```. (Do not forget to remove any older conda setup)

        . /software/2020/software/mamba/22.11.1-4/etc/profile.d/mamba.sh
        . /software/2020/software/mamba/22.11.1-4/etc/profile.d/conda.sh

## set up environment (CBE with gpu using MAMBA)
```
conda activate base
mamba create -n pt-gpu -c pytorch -c conda-forge -c default --file=env/env-pytorch-gpu-mamba.yml
conda activate pytorch-gpu-mamba
conda install -c conda-forge python-xxhash
conda install -c anaconda lz4
conda install -c conda-forge uproot4
```
## set up environment with conda
```
conda env create --file=env/env-pt.yml
```
