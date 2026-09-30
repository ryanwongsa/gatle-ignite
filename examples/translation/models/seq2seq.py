"""Encoder-decoder transformer: with tgt_in it teacher-forces, without it greedy-decodes.

Decoding inside forward keeps the default eval_step, and needs no DDP `.module` unwrapping.
"""

import torch
import torch.nn as nn

from examples.translation.dataloaders.data_utils.vocab import (
    BOS_ID,
    EOS_ID,
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    PAD_ID,
    TGT_VOCAB,
)


class Model(nn.Module):
    def __init__(
        self,
        in_dim=32,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        tgt_vocab=TGT_VOCAB,
        max_decode_len=MAX_TGT_LEN,
    ):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.max_decode_len = max_decode_len

        self.src_proj = nn.Linear(in_dim, d_model)
        self.src_pos = nn.Embedding(MAX_SRC_LEN + 1, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_pos = nn.Embedding(MAX_TGT_LEN + 1, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src_feats, src_pad_mask):
        t = src_feats.shape[1]
        pos = torch.arange(t, device=src_feats.device)
        x = self.src_proj(src_feats) + self.src_pos(pos).unsqueeze(0)
        return self.transformer.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(self, memory, src_pad_mask, tgt_in):
        u = tgt_in.shape[1]
        pos = torch.arange(u, device=tgt_in.device)
        y = self.tgt_emb(tgt_in) + self.tgt_pos(pos).unsqueeze(0)
        # Bool, not the float mask from generate_square_subsequent_mask: torch deprecates
        # mixing a float attn_mask with a bool key_padding_mask. True == masked out.
        causal = torch.triu(torch.ones(u, u, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.transformer.decoder(
            y,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_in == PAD_ID),
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out(h)

    @torch.no_grad()
    def greedy_decode(self, memory, src_pad_mask):
        bs = memory.shape[0]
        device = memory.device
        toks = torch.full((bs, 1), BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        for _ in range(self.max_decode_len):
            logits = self.decode(memory, src_pad_mask, toks)[:, -1]  # (B, V)
            nxt = logits.argmax(dim=-1)
            # Pad a row once it has emitted EOS, so the tensor stays honest past its end.
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS_ID)
            if bool(done.all()):
                break
        return toks[:, 1:]  # drop the BOS we seeded with

    def forward(self, src_feats, src_pad_mask, tgt_in=None):
        memory = self.encode(src_feats, src_pad_mask)
        if tgt_in is None:
            return {"pred_tokens": self.greedy_decode(memory, src_pad_mask)}
        return {"logits": self.decode(memory, src_pad_mask, tgt_in)}
