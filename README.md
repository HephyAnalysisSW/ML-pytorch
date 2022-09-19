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


## Setup weaver, ParticleNet

```
git clone https://github.com/hqucms/weaver.git
conda create -n weaver python=3.7
conda activate weaver
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
pip install uproot3 awkward0 lz4 xxhash
pip install tables
pip install onnxruntime-gpu
pip install tensorboard
pip install torch
```
