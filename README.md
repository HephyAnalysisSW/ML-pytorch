# ML-pytorch

## set up environment (CBE with gpu using MAMBA)
```
conda activate /groups/hephy/cms/dietrich.liko/conda/envs/my-base
mamba create -n pt-gpu -c pytorch -c conda-forge -c default --file=env/env-pytorch-gpu-mamba.yml
```
## set up environment (local)
```
conda env create --file=env/env-pt.yml
```

