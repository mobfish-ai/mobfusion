import argparse
import yaml
import shlex
import tempfile
import time
from subprocess import run
from pathlib import Path


def run_task(cmd: str):
    with tempfile.TemporaryDirectory() as dir:
        print(f'[{time.strftime("%X")}] -->', dir)

        cmd += f' --output_dir {shlex.quote(str(Path(dir) / "output"))}'
        cmd += f' --logging_dir {shlex.quote(str(Path(dir) / "logs"))}'

        print(f'[{time.strftime("%X")}] -->', cmd)

        out_fp = str(Path(dir) / "output.txt")
        with open(out_fp, "w+") as out:
            run(cmd, shell=True, stdout=out, stderr=out)

        print(f'[{time.strftime("%X")}] --> finished')


def parse_dataset_config(conf: dict):
    if "type" not in conf:
        raise Exception("Unknown dataset type")

    cmd = ""
    if conf["type"] == "local":
        cmd += f' --instance_data_dir {shlex.quote(conf["path"])}'
    else:
        raise Exception(f"Unsupported dataset type: {conf['type']}")

    return cmd


def generate_command(args: argparse.Namespace):
    with open(args.file, "r") as f:
        conf = yaml.safe_load(f)

    for k in ["dataset", "name", "train", "script"]:
        if not k in conf:
            raise Exception(f"'{k}' not in config")

    script = conf["script"]
    cmd = parse_dataset_config(conf["dataset"])

    for k, v in conf["train"].items():
        if isinstance(v, str) and v != "":
            cmd += f" --{k} {shlex.quote(v)}"
        elif isinstance(v, bool) and v:
            cmd += f" --{k}"
        elif isinstance(v, int) and v != 0:
            cmd += f" --{k} {shlex.quote(str(v))}"
        elif isinstance(v, float) and v != 0.0:
            cmd += f" --{k} {shlex.quote(str(v))}"

    return f"accelerate launch --num_cpu_threads_per_process 2 {script}.py{cmd}"


def train(args: argparse.Namespace):
    cmd = generate_command(args)
    run_task(cmd)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", type=str, required=True)

    args, _ = parser.parse_known_args()

    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
