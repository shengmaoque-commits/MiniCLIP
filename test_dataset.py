import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.tokenizer import SimpleTokenizer
from datasets.flickr30k import (
    Flickr30KDataset,
    split_dataframe
)


df = pd.read_csv(
    "data/flickr30k/captions.csv"
)

train_df, val_df, test_df = split_dataframe(
    df,
    seed=42
)

tokenizer = SimpleTokenizer.load(
    "data/flickr30k/vocab.json",
    max_length=40
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = Flickr30KDataset(
    dataframe=train_df,
    image_root="data/flickr30k/images",
    tokenizer=tokenizer,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

batch = next(iter(loader))

print(
    "image:",
    batch["image"].shape
)

print(
    "input_ids:",
    batch["input_ids"].shape
)

print(
    "attention_mask:",
    batch["attention_mask"].shape
)

print(
    "caption:",
    batch["caption"]
)