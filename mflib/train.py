import shlex
import subprocess
from pathlib import Path
from mflib.webhook import send_text
import math


def run_task(cmd: str, dir: str, verbose=False, webhook: str = None):
    root = Path(dir)

    with open(root / "cmd.txt", "w") as f:
        f.write(cmd)

    if verbose:
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with open(root / "output.txt", "wb") as f:
            f.write(proc.stdout)

    else:
        with open(root / "output.txt", "wb") as f:
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=f,
            )

    if proc.returncode > 0 and webhook:
        send_text(webhook, f"Failed returncode: {proc.returncode} dir: {dir}", True)
    else:
        send_text(webhook, f"Task finished: {dir}", True)


# TODO: add lora/lycoris support
def convert_to_sd(dir: str):
    p = Path(dir)
    flags = f"--model_path {shlex.quote(str(p / 'outputs'))}"
    flags += (
        f' --checkpoint_path {shlex.quote(str(p / "outputs" / "model.safetensors"))}'
    )
    flags += " --use_safetensors"
    proc = subprocess.run(
        f"python pipeline/convert_to_sd.py {flags}",
        shell=True,
        capture_output=False,
    )

    if proc.returncode:
        raise Exception(
            f"Convert to Stable Diffusion format fail, return code: {proc.returncode}"
        )


def parse_optimizer_flags(conf: dict):
    if "optimizer" not in conf:
        return " --use_8bit_adam"

    opt = conf["optimizer"]
    flags = ""

    # TODO: full optimizer params supported
    if opt == "AdamW8bit":
        flags += " --use_8bit_adam"
    elif opt == "Lion8bit":
        flags += " --use_8bit_lion"
    else:
        raise Exception(f"Unsupported optimizer: {opt}")

    return flags


def parse_sample_flags(conf: dict):
    flags = ""

    if "prompt" not in conf:
        return flags

    # TODO: full sample params supported
    prompt = conf["prompt"].replace("%i", conf["instance_prompt"])
    flags += f" --validation_prompt {shlex.quote(prompt)}"

    sample_steps = conf["max_train_steps"]

    if "sample_every_steps" in conf:
        sample_steps = conf["sample_every_steps"]
    elif "sample_every_epochs" in conf:
        sample_steps = math.floor(
            conf["sample_every_steps"] / conf["sample_every_epochs"]
        )

    flags += f" --validation_steps {sample_steps}"

    return flags


def parse_flags(conf: dict, root: str):
    p = Path(root)

    flags = f"--pretrained_model_name_or_path {shlex.quote(conf['pretrained'])}"
    flags += f" --instance_data_dir {shlex.quote(conf['train_dataset'])}"
    flags += f" --output_dir {shlex.quote(str(p / 'outputs'))}"
    flags += f" --logging_dir {shlex.quote(str(p / 'logs'))}"
    flags += f" --sample_image_dir {shlex.quote(str(p / 'samples'))}"
    flags += f" --train_batch_size {shlex.quote(str(conf['batch_size']))}"
    flags += f" --gradient_accumulation_steps {shlex.quote(str(conf['gradient_accumulation_steps']))}"
    flags += f" --resolution {shlex.quote(str(conf['resolution']))}"
    flags += f" --caption_ext {shlex.quote(conf['caption_ext'])}"
    flags += f" --learning_rate {shlex.quote(conf['learning_rate'])}"
    flags += f" --lr_scheduler {shlex.quote(conf['lr_scheduler'])}"
    flags += f" --lr_warmup_steps {shlex.quote(str(conf['lr_warmup_steps']))}"
    flags += f" --max_train_steps {shlex.quote(str(conf['max_train_steps']))}"
    flags += f" --dataloader_num_workers {shlex.quote(str(conf['dataloader_workers']))}"

    flags += f" --instance_prompt {shlex.quote(conf['instance_prompt'])}"
    flags += parse_optimizer_flags(conf)

    if "xformers" in conf and conf["xformers"]:
        flags += " --enable_xformers_memory_efficient_attention"

    flags += parse_sample_flags(conf)

    if "webhook" in conf:
        flags += f' --webhook {shlex.quote(conf["webhook"])}'

    return flags
