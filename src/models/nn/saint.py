import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# https://github.com/arshadshk/SAINT-pytorch/blob/main/saint.py
class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        self.embd_cat = nn.Embedding(total_cat + 1, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex, in_cat, first_block=True):
        device = in_ex.device

        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = self.embd_cat(in_cat)
            # in_pos = self.embd_pos( in_pos )
            # combining the embedings
            out = in_ex + in_cat   # + in_pos
        else:
            out = in_ex

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """
    def __init__(self, dim_model, total_in, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.embd_in = nn.Embedding(total_in, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.ffn_en = Feed_Forward_block(dim_model)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, en_out, first_block=True):
        device = in_in.device

        if first_block:
            in_in = self.embd_in(in_in)

            out = in_in
        else:
            out = in_in

        in_pos = get_pos(self.seq_len, device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = skip_out + out

        en_out = en_out.permute(1, 0, 2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n, device=device))
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len, device):
    mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)).to(device)
    return mask


def get_pos(seq_len, device):
    # use sine positional embeddinds
    # return torch.arange(seq_len, device=device).unsqueeze(0)
    return torch.arange(seq_len, device=device).unsqueeze(0)


class SAINT(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_in, heads_de, seq_len):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(Encoder_block(dim_model, heads_en, total_ex, total_cat, seq_len), num_en)
        self.decoder = get_clones(Decoder_block(dim_model, total_in, heads_de, seq_len), num_de)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, feat):
        in_ex = feat['in_ex']
        in_cat = feat['in_cat']
        in_in = feat['in_de']

        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, first_block=first_block)
            in_cat = in_ex

        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](in_in, en_out=in_ex, first_block=first_block)

        # in_in = torch.sigmoid(self.out(in_in))
        in_in = self.out(in_in)

        return in_in.squeeze(-1)
