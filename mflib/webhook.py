import requests
import tempfile
from PIL.Image import Image
from pathlib import Path
import threading


# TODO: handle exception
def send_text(url: str, text: str, signal=False):
    if signal:
        requests.post(url, json={"content": text})
    else:
        threading.Thread(
            target=lambda: requests.post(url, json={"content": text}),
            daemon=True,
        ).start()


def send_file(url: str, files: list[str]):
    def target():
        file_list = {f"files[{i}]": open(name, "rb") for i, name in enumerate(files)}
        try:
            requests.post(url, files=file_list)
        finally:
            for f in file_list.values():
                f.close()

    threading.Thread(target=target, daemon=True).start()


def send_samples(url: str, imgs: list[Image]):
    files = []
    with tempfile.TemporaryDirectory() as p:
        for i, img in enumerate(imgs):
            fp = str(Path(p) / f"{i}.png")
            img.save(fp)
            files.append(fp)

        send_file(url, files)
