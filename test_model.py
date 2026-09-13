import torch

from models.miniclip import MiniCLIP


batch_size = 4

vocab_size = 10000

max_length = 40


model = MiniCLIP(
    vocab_size=vocab_size,
    max_length=max_length,

    vision_feature_dim=512,
    text_hidden_dim=512,

    embed_dim=256,

    text_num_layers=4,
    text_num_heads=8,
    text_ff_dim=2048,

    temperature=0.07
)


images = torch.randn(
    batch_size,
    3,
    224,
    224
)


input_ids = torch.randint(
    0,
    vocab_size,
    (
        batch_size,
        max_length
    )
)


attention_mask = torch.ones(
    batch_size,
    max_length,
    dtype=torch.long
)


outputs = model(
    images,
    input_ids,
    attention_mask
)


print(
    "image_embeddings:",
    outputs["image_embeddings"].shape
)

print(
    "text_embeddings:",
    outputs["text_embeddings"].shape
)

print(
    "logits:",
    outputs["logits"].shape
)

print(
    "logit_scale:",
    outputs["logit_scale"]
)