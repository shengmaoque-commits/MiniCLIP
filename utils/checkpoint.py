import os
import torch


class CheckpointManager:

    def __init__(
        self,
        save_dir,
    ):
        self.save_dir = save_dir

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        self.best_val_loss = float("inf")

    def save_last(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        val_loss,
        config,
    ):

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }

        if scheduler is not None:
            checkpoint["scheduler"] = \
                scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler"] = \
                scaler.state_dict()

        path = os.path.join(
            self.save_dir,
            "last.pth"
        )

        torch.save(
            checkpoint,
            path
        )

    def save_best(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        val_loss,
        config,
    ):

        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }

        if scheduler is not None:
            checkpoint["scheduler"] = \
                scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler"] = \
                scaler.state_dict()

        path = os.path.join(
            self.save_dir,
            "best.pth"
        )

        torch.save(
            checkpoint,
            path
        )

        return True