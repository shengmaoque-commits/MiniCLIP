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
from datasets.split import split_dataframe


csv_path = "data/Images/results.csv"

df = pd.read_csv(
    csv_path,
    sep="|",
    skipinitialspace=True
)

# 清理列名两边的空格
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())


# Flickr30K 原始列名通常是：
# image_name
# comment_number
# comment

df = df.rename(
    columns={
        "image_name": "image",
        "comment": "caption"
    }
)

# 再清理字符串
df["image"] = df["image"].astype(str).str.strip()
df["caption"] = df["caption"].astype(str).str.strip()

# 去掉空 caption
df = df[
    df["caption"].notna()
].reset_index(drop=True)


print("\nAfter rename:")
print(df.columns.tolist())

print(df.head())


# ============================
# Split
# ============================

train_df, val_df, test_df = split_dataframe(
    df,
    train_ratio=0.9,
    val_ratio=0.05,
    seed=42
)


print(
    f"\nTrain images: {train_df['image'].nunique()}"
)

print(
    f"Val images: {val_df['image'].nunique()}"
)

print(
    f"Test images: {test_df['image'].nunique()}"
)


# ============================
# Build vocabulary
# ============================

tokenizer = SimpleTokenizer(
    max_length=40
)

tokenizer.build_vocab(
    train_df["caption"].tolist(),
    min_freq=2,
    max_vocab_size=20000
)


# ============================
# Save
# ============================

vocab_path = "data/Images/vocab.json"

tokenizer.save(
    vocab_path
)

print(
    f"\nVocabulary saved to: {vocab_path}"
)