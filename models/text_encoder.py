import torch
import torch.nn as nn


class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_length=40,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=pad_id,
        )

        # 2. Position Embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_dim,
        )

        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 4. 多层 Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 5. 最后 LayerNorm
        self.final_norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        input_ids,
        attention_mask=None
    ):

        batch_size, seq_length = input_ids.shape

        # -----------------------
        # Token embedding
        # -----------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        # [B, L]
        positions = torch.arange(
            seq_length,
            device=input_ids.device
        )

        # [L] → [B, L]
        positions = positions.unsqueeze(0).expand(
            batch_size,
            seq_length
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Token + Position
        x = (
            token_embeddings
            + position_embeddings
        )

        # -----------------------
        # Padding mask
        # -----------------------

        if attention_mask is not None:

            # Transformer:
            # True 表示这个位置不要关注

            padding_mask = (
                attention_mask == 0
            )

        else:

            padding_mask = None

        # -----------------------
        # Transformer
        # -----------------------

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.final_norm(x)

        # -----------------------
        # 取 CLS token
        # -----------------------

        cls_feature = x[:, 0, :]

        return cls_feature