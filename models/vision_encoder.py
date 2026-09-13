import torch
import torch.nn as nn

from torchvision.models import resnet18


class VisionEncoder(nn.Module):

    def __init__(
        self,
        pretrained=False,
        output_dim=512
    ):
        super().__init__()

        if pretrained:
            from torchvision.models import ResNet18_Weights

            backbone = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            backbone = resnet18(
                weights=None
            )

        # 去掉最后的分类层 fc
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.output_dim = output_dim

    def forward(self, images):

        features = self.backbone(images)

        return features