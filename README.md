# mobfusion

**⚠️ WIP**

CAUTION, this project currently is experimental!

## Prepare

Currently you have to prepare pretrained models and datasets manually.

Download Huggingface models via git lfs, eg.

```shell
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

## Usage

```shell
# (optional) prepare your python environment,
# recommanded conda with miniforge
pip install -U -r requirements_dev.txt

# (Optional) if bitsandbytes caused error,
# you should add this env
export LD_LIBRARY_PATH=/home/xxx/miniforge3/envs/xxx/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_LATH

# step 1. prepare your dataset
# eg.
git clone https://yourdataset /path/to/dataset

# step 2. prepare your config file: train.yaml
cp train.yaml your_conf.yaml
vim your_conf.yaml

# step 3. start your training
./mobfusion -f your_conf.yaml
```

## TODO

- [ ] Add `dadaptation` support
- [ ] Add Lora/Lycoris support
