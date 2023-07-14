from diffusers import StableDiffusionPipeline
import argparse


def predict(prompt: str, output: str, model: str):
    pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda")
    img = pipe(prompt, num_inference_steps=20).images[0]
    img.save(output)


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
