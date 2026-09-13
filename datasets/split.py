import numpy as np


def split_dataframe(
    dataframe,
    train_ratio=0.9,
    val_ratio=0.05,
    seed=42
):
    unique_images = dataframe["image"].unique()
    # random number generator
    rng = np.random.default_rng(seed)

    rng.shuffle(unique_images)

    num_images = len(unique_images)

    train_end = int(
        num_images * train_ratio
    )

    val_end = int(
        num_images * (train_ratio + val_ratio)
    )

    train_images = unique_images[:train_end]

    val_images = unique_images[
        train_end:val_end
    ]

    test_images = unique_images[
        val_end:
    ]

    train_df = dataframe[
        dataframe["image"].isin(train_images)
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["image"].isin(val_images)
    ].reset_index(drop=True)

    test_df = dataframe[
        dataframe["image"].isin(test_images)
    ].reset_index(drop=True)

    return train_df, val_df, test_df