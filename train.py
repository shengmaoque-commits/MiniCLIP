import os
import random
import argparse

import yaml
import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from datasets.tokenizer import SimpleTokenizer
from datasets.flickr30k import Flickr30KDataset
from datasets.split import split_dataframe

from models.miniclip import MiniCLIP

from utils.loss import (
    clip_loss,
    contrastive_accuracy
)

from utils.checkpoint import CheckpointManager

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

def build_dataloaders(
    config,
    tokenizer
):

    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    root = dataset_cfg["root"]

    csv_path = os.path.join(
        root,
        dataset_cfg["captions_file"]
    )

    image_root = os.path.join(
        root,
        dataset_cfg["image_dir"]
    )

    dataframe = pd.read_csv(
        csv_path
    )

    train_df, val_df, test_df = split_dataframe(
        dataframe,
        train_ratio=dataset_cfg["train_ratio"],
        val_ratio=dataset_cfg["val_ratio"],
        seed=config["experiment"]["seed"]
    )

    print(
        f"Train images: "
        f"{train_df['image'].nunique()}"
    )

    print(
        f"Val images: "
        f"{val_df['image'].nunique()}"
    )

    print(
        f"Test images: "
        f"{test_df['image'].nunique()}"
    )

    image_size = dataset_cfg["image_size"]

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            image_size + 32
        ),

        transforms.CenterCrop(
            image_size
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    train_dataset = Flickr30KDataset(
        dataframe=train_df,
        image_root=image_root,
        tokenizer=tokenizer,
        transform=train_transform,
        random_caption=True,
    )

    val_dataset = Flickr30KDataset(
        dataframe=val_df,
        image_root=image_root,
        tokenizer=tokenizer,
        transform=val_transform,
        random_caption=False,
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=train_cfg["batch_size"],

        shuffle=True,

        num_workers=train_cfg["num_workers"],

        pin_memory=True,

        drop_last=True,

        persistent_workers=(
            train_cfg["num_workers"] > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=eval_cfg["batch_size"],

        shuffle=False,

        num_workers=train_cfg["num_workers"],

        pin_memory=True,

        drop_last=False,

        persistent_workers=(
            train_cfg["num_workers"] > 0
        ),
    )

    return (
        train_loader,
        val_loader
    )


def build_model(
    config,
    tokenizer
):

    model_cfg = config["model"]

    text_cfg = model_cfg["text"]

    model = MiniCLIP(

        vocab_size=len(tokenizer),

        max_length=config[
            "dataset"
        ]["max_text_length"],

        vision_feature_dim=model_cfg[
            "image_feature_dim"
        ],

        text_hidden_dim=text_cfg[
            "hidden_dim"
        ],

        embed_dim=model_cfg[
            "embed_dim"
        ],

        text_num_layers=text_cfg[
            "num_layers"
        ],

        text_num_heads=text_cfg[
            "num_heads"
        ],

        text_ff_dim=text_cfg[
            "ff_dim"
        ],

        text_dropout=text_cfg[
            "dropout"
        ],

        temperature=model_cfg[
            "temperature"
        ],

        pretrained_vision=model_cfg[
            "pretrained_vision"
        ],

        pad_id=tokenizer.pad_id,
    )

    return model
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    config,
):

    model.train()

    total_loss = 0.0

    total_i2t = 0.0

    total_t2i = 0.0

    total_acc_i2t = 0.0

    total_acc_t2i = 0.0

    use_amp = (
        config["train"]["amp"]
        and
        device.type == "cuda"
    )

    progress_bar = tqdm(
        loader,
        desc=f"Train {epoch}"
    )

    for batch in progress_bar:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ==========================
        # Forward
        # ==========================

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images,
                    input_ids,
                    attention_mask
                )

                loss_dict = clip_loss(
                    outputs["logits"]
                )

                loss = loss_dict["loss"]

        else:

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            loss_dict = clip_loss(
                outputs["logits"]
            )

            loss = loss_dict["loss"]

        # ==========================
        # Backward
        # ==========================

        if use_amp:

            scaler.scale(
                loss
            ).backward()

            # 如果需要 gradient clipping
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config[
                    "train"
                ]["grad_clip"]
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config[
                    "train"
                ]["grad_clip"]
            )

            optimizer.step()

        # ==========================
        # Metrics
        # ==========================

        acc_dict = contrastive_accuracy(
            outputs["logits"].detach()
        )

        total_loss += \
            loss.item()

        total_i2t += \
            loss_dict["loss_i2t"].item()

        total_t2i += \
            loss_dict["loss_t2i"].item()

        total_acc_i2t += \
            acc_dict["acc_i2t"].item()

        total_acc_t2i += \
            acc_dict["acc_t2i"].item()

        num_steps = (
            progress_bar.n + 1
        )

        progress_bar.set_postfix({

            "loss":
                f"{total_loss / num_steps:.4f}",

            "I2T":
                f"{total_acc_i2t / num_steps:.3f}",

            "T2I":
                f"{total_acc_t2i / num_steps:.3f}",

            "scale":
                f"{outputs['logit_scale'].item():.2f}",
        })

    num_batches = len(loader)

    return {

        "loss":
            total_loss / num_batches,

        "loss_i2t":
            total_i2t / num_batches,

        "loss_t2i":
            total_t2i / num_batches,

        "acc_i2t":
            total_acc_i2t / num_batches,

        "acc_t2i":
            total_acc_t2i / num_batches,
    }

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    epoch,
):

    model.eval()

    total_loss = 0.0

    total_i2t = 0.0

    total_t2i = 0.0

    total_acc_i2t = 0.0

    total_acc_t2i = 0.0

    progress_bar = tqdm(
        loader,
        desc=f"Val   {epoch}"
    )

    for batch in progress_bar:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True
        )

        outputs = model(
            images,
            input_ids,
            attention_mask
        )

        loss_dict = clip_loss(
            outputs["logits"]
        )

        acc_dict = contrastive_accuracy(
            outputs["logits"]
        )

        total_loss += \
            loss_dict["loss"].item()

        total_i2t += \
            loss_dict["loss_i2t"].item()

        total_t2i += \
            loss_dict["loss_t2i"].item()

        total_acc_i2t += \
            acc_dict["acc_i2t"].item()

        total_acc_t2i += \
            acc_dict["acc_t2i"].item()

        num_steps = (
            progress_bar.n + 1
        )

        progress_bar.set_postfix({

            "loss":
                f"{total_loss / num_steps:.4f}",

            "I2T":
                f"{total_acc_i2t / num_steps:.3f}",

            "T2I":
                f"{total_acc_t2i / num_steps:.3f}",
        })

    num_batches = len(loader)

    return {

        "loss":
            total_loss / num_batches,

        "loss_i2t":
            total_i2t / num_batches,

        "loss_t2i":
            total_t2i / num_batches,

        "acc_i2t":
            total_acc_i2t / num_batches,

        "acc_t2i":
            total_acc_t2i / num_batches,
    }

def main(args):

    # ==============================
    # Config
    # ==============================

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    seed = config[
        "experiment"
    ]["seed"]

    set_seed(seed)

    # ==============================
    # Device
    # ==============================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ==============================
    # Tokenizer
    # ==============================

    vocab_path = os.path.join(
        config["dataset"]["root"],
        "vocab.json"
    )

    tokenizer = SimpleTokenizer.load(
        vocab_path,
        max_length=config[
            "dataset"
        ]["max_text_length"]
    )

    print(
        f"Vocabulary size: {len(tokenizer)}"
    )

    # ==============================
    # Dataset
    # ==============================

    train_loader, val_loader = \
        build_dataloaders(
            config,
            tokenizer
        )

    # ==============================
    # Model
    # ==============================

    model = build_model(
        config,
        tokenizer
    )

    model = model.to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{num_params / 1e6:.2f} M"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params / 1e6:.2f} M"
    )
    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config[
            "train"
        ]["lr"],

        weight_decay=config[
            "train"
        ]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config[
            "train"
        ]["epochs"]
    )
    use_amp = (
        config["train"]["amp"]
        and
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )
    experiment_name = config[
        "experiment"
    ]["name"]

    save_dir = os.path.join(

        config[
            "train"
        ]["save_dir"],

        experiment_name
    )

    checkpoint_manager = \
        CheckpointManager(
            save_dir
        )

    epochs = config[
        "train"
    ]["epochs"]

    for epoch in range(
        1,
        epochs + 1
    ):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"LR: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print(
            f"{'=' * 60}"
        )

        # ==========================
        # Train
        # ==========================

        train_stats = train_one_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            epoch=epoch,

            config=config,
        )

        # ==========================
        # Validation
        # ==========================

        val_stats = validate(

            model=model,

            loader=val_loader,

            device=device,

            epoch=epoch,
        )

        scheduler.step()

        # ==========================
        # Print
        # ==========================

        print(
            "\nTrain:"
        )

        print(
            f"Loss: "
            f"{train_stats['loss']:.4f}"
        )

        print(
            f"I2T Acc: "
            f"{train_stats['acc_i2t']:.4f}"
        )

        print(
            f"T2I Acc: "
            f"{train_stats['acc_t2i']:.4f}"
        )

        print(
            "\nValidation:"
        )

        print(
            f"Loss: "
            f"{val_stats['loss']:.4f}"
        )

        print(
            f"I2T Acc: "
            f"{val_stats['acc_i2t']:.4f}"
        )

        print(
            f"T2I Acc: "
            f"{val_stats['acc_t2i']:.4f}"
        )

        # ==========================
        # Save Last
        # ==========================

        checkpoint_manager.save_last(

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            epoch=epoch,

            val_loss=val_stats["loss"],

            config=config,
        )

        # ==========================
        # Save Best
        # ==========================

        is_best = \
            checkpoint_manager.save_best(

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                scaler=scaler,

                epoch=epoch,

                val_loss=val_stats["loss"],

                config=config,
            )

        if is_best:

            print(
                f"✓ New best checkpoint "
                f"saved."
            )
    print(
        "\nTraining finished."
    )

    print(
        f"Checkpoints saved to: "
        f"{save_dir}"
    )    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml"
    )

    args = parser.parse_args()

    main(args)