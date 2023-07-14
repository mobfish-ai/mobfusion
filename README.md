# mobfusion

**⚠️ WIP**

CAUTION, this project currently in experimental!

## Usage

```shell
# (optional) prepare your python environment,
# recommanded conda with miniforge
pip install -U -r requirements_dev.txt

# (Optional) if bitsandbytes caused error,
# you should add this env
export LD_LIBRARY_PATH=/home/xxx/miniforge3/envs/xxx/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_LATH

# step 1. prepare your dataset
# step 2. prepare your config file: train.yaml
# step 3. start your training
python train.py -f train.yaml
```