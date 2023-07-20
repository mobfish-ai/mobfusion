from pathlib import Path
from PIL.Image import Image
import tempfile
from mflib.webhook import send_file

_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def list_dir_imgs(path: str):
    p = Path(path)
    imgs = []

    for i in p.iterdir():
        for ext in _IMAGE_EXTS:
            if str(i).endswith(ext):
                imgs.append(i)

    return imgs


def send_webhook_samples(url: str, imgs: list[Image]):
    files = []
    with tempfile.TemporaryDirectory() as p:
        for i, img in enumerate(imgs):
            fp = str(Path(p) / f"{i}.png")
            img.save(fp)
            files.append(fp)

        send_file(url, files)
