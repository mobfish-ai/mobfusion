from pathlib import Path
import time
import random
import os
from PIL.Image import Image

_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def list_dir_imgs(path: str):
    p = Path(path)
    imgs = []

    for i in p.iterdir():
        for ext in _IMAGE_EXTS:
            if str(i).endswith(ext):
                imgs.append(i)

    return imgs


def save_sample_images(
    dir: str,
    images: list[Image],
    epoch: int,
    steps: int,
) -> list[str]:
    p = Path(dir)

    files = []
    for i, img in enumerate(images):
        fn = str(p / f"epoch_{epoch}_steps_{steps}_{str(i).zfill(3)}.png")
        img.save(fn)
        files.append(fn)

    return files


def printt(*args):
    print(f'[{time.strftime("%X")}]', *args)


def make_task_dir(name: str, workspace: str):
    root = Path(workspace)
    ts = time.strftime(r"%y%m%d%H%M%S")
    rd = random.randbytes(4).hex()
    p = str(root / name.replace(r"%t", ts).replace(r"%r", rd))
    os.makedirs(p, exist_ok=True)
    return p


def conf_reduce_to_str(conf: dict):
    summary = f'Task name: {conf["name"]}'

    return summary
