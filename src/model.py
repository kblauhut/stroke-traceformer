import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_size):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_size = embed_size

        self.proj = nn.Conv2d(in_channels=3, out_channels=embed_size,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch, embed_size, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_size)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_length, embed_size)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)  # Apply positional encoding

class TransformerModel(nn.Module):
    def __init__(self, tgt_vocab_size, config, device):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = device
        self.tgt_vocab_size = tgt_vocab_size

        # Vision Transformer Encoder
        self.patch_embed = PatchEmbedding(config.img_size, config.patch_size, config.embed_size).to(device)
        self.src_positional_encoding = PositionalEncoding(config.embed_size, self.patch_embed.num_patches).to(device)

        # SVG Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.embed_size).to(device)
        self.tgt_positional_encoding = PositionalEncoding(config.embed_size, config.context_length).to(device)

        self.transformer = nn.Transformer(
            d_model=config.embed_size,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dropout=config.dropout,
            bias=False,
            activation="gelu"
        ).to(device)

        # Output Projection
        self.fc_out = nn.Linear(config.embed_size, tgt_vocab_size).to(device)

        # Shared weights for embeddings
        self.tgt_embedding.weight = self.fc_out.weight

    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, 3, img_size, img_size) - Input image
            tgt: (batch_size, seq_len) - Target SVG token sequence
        Returns:
            Logits for SVG tokens (batch_size, seq_len, tgt_vocab_size)
        """
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        # Encode image patches
        src_emb = self.src_positional_encoding(self.patch_embed(src))
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, embed_size)


        # Encode target sequence (SVG tokens)
        tgt_emb = self.tgt_positional_encoding(self.tgt_embedding(tgt))
        tgt_emb = tgt_emb.permute(1, 0, 2)  # (seq_len, batch, embed_size)

        # Generate masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(self.device)
        tgt_key_padding_mask = (tgt == 0)  # Mask padding tokens

        # Run through Transformer
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary space
        output = self.fc_out(output.permute(1, 0, 2))  # (batch, seq_len, vocab_size)
        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
