import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.projection import ProjectionHead


class MiniCLIP(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_length=40,

        vision_feature_dim=512,
        text_hidden_dim=512,

        embed_dim=256,

        text_num_layers=4,
        text_num_heads=8,
        text_ff_dim=2048,
        text_dropout=0.1,

        temperature=0.07,

        pretrained_vision=False,
        pad_id=0,
    ):
        super().__init__()

        # ========================
        # Image Encoder
        # ========================

        self.vision_encoder = VisionEncoder(
            pretrained=pretrained_vision,
            output_dim=vision_feature_dim
        )

        # ========================
        # Text Encoder
        # ========================

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            ff_dim=text_ff_dim,
            dropout=text_dropout,
            pad_id=pad_id,
        )

        # ========================
        # Projection
        # ========================

        self.image_projection = ProjectionHead(
            input_dim=vision_feature_dim,
            output_dim=embed_dim
        )

        self.text_projection = ProjectionHead(
            input_dim=text_hidden_dim,
            output_dim=embed_dim
        )

        # ========================
        # Learnable temperature
        # ========================

        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / temperature)
            )
        )

    def encode_image(
        self,
        images
    ):

        image_features = self.vision_encoder(
            images
        )

        image_embeddings = self.image_projection(
            image_features
        )

        image_embeddings = F.normalize(
            image_embeddings,
            dim=-1
        )

        return image_embeddings

    def encode_text(
        self,
        input_ids,
        attention_mask
    ):

        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embeddings = self.text_projection(
            text_features
        )

        text_embeddings = F.normalize(
            text_embeddings,
            dim=-1
        )

        return text_embeddings

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):

        image_embeddings = self.encode_image(
            images
        )

        text_embeddings = self.encode_text(
            input_ids,
            attention_mask
        )

        # 避免训练过程中无限变大
        logit_scale = self.logit_scale.exp()

        logit_scale = torch.clamp(
            logit_scale,
            max=100.0
        )

        logits = (
            logit_scale
            * image_embeddings
            @ text_embeddings.t()
        )

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits": logits,
            "logit_scale": logit_scale,
        }