from diffusers import StableDiffusionPipeline
import argparse
import tempfile
from mflib.webhook import send_file
import os
from pathlib import Path


def predict(prompt: str, output: str, model: str):
    pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda")
    # gen = torch.Generator(pipe.device).manual_seed(123)
    imgs = pipe(prompt, num_inference_steps=20, num_images_per_prompt=4).images

    with tempfile.TemporaryDirectory() as p:
        files = []
        dir = Path(p)

        for i, img in enumerate(imgs):
            f = str(dir / f"{i}.png")
            print(f)
            img.save(f)
            files.append(f)

        send_file(os.environ["WEBHOOK_URL"], files)

    # if output is None:
    #     with tempfile.NamedTemporaryFile(suffix=".png") as f:
    #         print(f.name)
    #         img.save(f.name)
    #         send_file(os.environ["WEBHOOK_URL"], [f.name])
    # else:
    #     img.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--output", type=str)

    args, _ = parser.parse_known_args()
    predict(
        prompt=args.prompt,
        output=args.output,
        model=args.model,
    )
