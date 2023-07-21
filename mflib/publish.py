import os
from pathlib import Path
import subprocess
import shlex
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from mflib.util import printt


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipeline, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipeline, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- dreambooth
inference: true
---
    """
    model_card = f"""
# DreamBooth - {repo_id}

This is a dreambooth model derived from {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/).
You can find some example images in the following. \n
{img_str}

DreamBooth for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def _shell(cmd: str, cwd=None):
    printt(cmd)
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True)


def push_to_git(
    dir: str,
    git_url: str,
    commit_message: str,
    user: str = None,
    email: str = None,
):
    if not (Path(dir) / ".git").exists():
        _shell(f"git init -b main {shlex.quote(dir)}", dir)

    _shell("git lfs track *.safetensors **/*.safetensors", dir)
    _shell("git add .", dir)

    if user is not None:
        _shell(f"git config user.name {shlex.quote(user)}", dir)

    if email is not None:
        _shell(f"git config user.email {shlex.quote(email)}", dir)

    _shell(f"git commit -m {shlex.quote(commit_message)}", dir)
    _shell(f"git push {shlex.quote(git_url)} main", dir)


# TODO: reduce conf for commit message
def format_commit_message(conf: dict):
    return "Training result"
