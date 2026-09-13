import os
import sys
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from datasets.tokenizer import SimpleTokenizer
from datasets.flickr30k import split_dataframe


csv_path = "data/flickr30k/captions.csv"

df = pd.read_csv(csv_path)

train_df, val_df, test_df = split_dataframe(
    df,
    train_ratio=0.9,
    val_ratio=0.05,
    seed=42
)

tokenizer = SimpleTokenizer(
    max_length=40
)

tokenizer.build_vocab(
    train_df["caption"].tolist(),
    min_freq=2,
    max_vocab_size=20000
)

os.makedirs(
    "data/flickr30k",
    exist_ok=True
)

tokenizer.save(
    "data/flickr30k/vocab.json"
)

print("Vocabulary saved.")