import torch
import torch.nn.functional as F


def clip_loss(logits):
    """
    Args:
        logits:
            [B, B]

            logits[i, j] 表示：
            第 i 张图片和第 j 条文本的匹配程度

    Returns:
        loss
        loss_i2t
        loss_t2i
    """

    batch_size = logits.size(0)

    labels = torch.arange(
        batch_size,
        device=logits.device
    )

    # Image -> Text
    loss_i2t = F.cross_entropy(
        logits,
        labels
    )

    # Text -> Image
    loss_t2i = F.cross_entropy(
        logits.t(),
        labels
    )

    loss = (
        loss_i2t + loss_t2i
    ) / 2.0

    return {
        "loss": loss,
        "loss_i2t": loss_i2t,
        "loss_t2i": loss_t2i,
    }


@torch.no_grad()
def contrastive_accuracy(logits):

    batch_size = logits.size(0)

    labels = torch.arange(
        batch_size,
        device=logits.device
    )

    # Image -> Text
    pred_i2t = logits.argmax(dim=1)

    acc_i2t = (
        pred_i2t == labels
    ).float().mean()

    # Text -> Image
    pred_t2i = logits.argmax(dim=0)

    acc_t2i = (
        pred_t2i == labels
    ).float().mean()

    return {
        "acc_i2t": acc_i2t,
        "acc_t2i": acc_t2i,
    }

