from mflib.webhook import send_text
import time


def start_monitor(url: str):
    while True:
        send_text(url, "GPU temp: 100C")

        time.sleep(10)
