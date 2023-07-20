from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from os.path import splitext

from mflib.util import list_dir_imgs


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def read_caption(p: str, ext=".txt"):
    base, _ = splitext(p)
    fp = Path(base + ext)
    if fp.exists():
        with open(fp, "r") as f:
            return f.readline()

    fp = Path(p + ext)
    if fp.exists():
        with open(fp, "r") as f:
            return f.readline()

    return None


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        caption_ext=".txt",
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = (
            instance_prompt_encoder_hidden_states
        )
        self.tokenizer_max_length = tokenizer_max_length
        self.prompt_map = {}
        self.class_map = {}

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        self.instance_images_path = list_dir_imgs(instance_data_root)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        for i in self.instance_images_path:
            fp = str(i)
            if caption := read_caption(fp, caption_ext) is not None:
                self.prompt_map[fp] = caption

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list_dir_imgs(class_data_root)
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt

            for i in self.class_images_path:
                fp = str(i)
                if caption := read_caption(fp, caption_ext) is not None:
                    self.class_map[fp] = caption
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        current = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(current)
        instance_image = exif_transpose(instance_image)

        instance_prompt = (
            self.prompt_map[current]
            if current in self.prompt_map
            else self.instance_prompt
        )

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer,
                instance_prompt,
                tokenizer_max_length=self.tokenizer_max_length,
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            current = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(current)
            class_image = exif_transpose(class_image)

            class_prompt = (
                self.class_map[current]
                if current in self.class_map
                else self.class_prompt
            )

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.instance_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.instance_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer,
                    class_prompt,
                    tokenizer_max_length=self.tokenizer_max_length,
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example
