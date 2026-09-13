import os
import random
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class Flickr30KDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_root,
        tokenizer,
        transform=None,
        random_caption=True,
    ):
        super().__init__()

        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.random_caption = random_caption

        # 按图片分组
        grouped = dataframe.groupby("image_name")["comment"].apply(list)

        self.image_names = list(grouped.index)

        self.captions = {
            image_name: captions
            for image_name, captions in grouped.items()
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = os.path.join(
            self.image_root,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.captions[image_name]

        if self.random_caption:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        token_ids, attention_mask = \
            self.tokenizer.encode(caption)

        token_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.long
        )

        return {
            "image": image,
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "image_name": image_name,
        }

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])