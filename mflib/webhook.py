import requests
import tempfile
from PIL.Image import Image
from pathlib import Path
import threading


def __send_discord(url: str, data, is_file=False):
    try:
        if is_file:
            requests.post(url, files=data, timeout=3.0)
        else:
            requests.post(url, json=data, timeout=3.0)
    except:
        pass
    finally:
        if is_file:
            for f in data.values():
                f.close()


def send_text(url: str, text: str, signal=False):
    if signal:
        __send_discord(url, {"content": text})
    else:
        threading.Thread(
            target=__send_discord,
            daemon=True,
            args=(url, {"content": text}),
        ).start()


def send_file(url: str, files: list[str]):
    file_list = {f"files[{i}]": open(name, "rb") for i, name in enumerate(files)}

    threading.Thread(
        target=__send_discord,
        daemon=True,
        args=(url, file_list, True),
    ).start()


def send_samples(url: str, imgs: list[Image]):
    with tempfile.TemporaryDirectory() as p:
        files = []
        for i, img in enumerate(imgs):
            fp = str(Path(p) / f"{i}.png")
            img.save(fp)
            files.append(fp)

        send_file(url, files)
