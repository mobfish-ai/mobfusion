import requests


def send_text(url: str, text: str):
    requests.post(url, json={"content": text})


def send_file(url: str, files: list[str]):
    file_list = {f"files[{i}]": open(name, "rb") for i, name in enumerate(files)}
    try:
        requests.post(url, files=file_list)
    finally:
        for f in file_list.values():
            f.close()
