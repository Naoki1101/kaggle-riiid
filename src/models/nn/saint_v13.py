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

    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_tg, seq_len):
        super().__init__()
        self.seq_len = seq_len - 1
        self.total_cat = total_cat
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        # self.embd_cat = nn.Embedding(total_cat + 1, embedding_dim=dim_model)
        self.embd_tg = nn.Embedding(total_tg + 1, embedding_dim=dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)
        self.pos_norm = nn.LayerNorm(dim_model, eps=1.0e-12)
        self.dt_fc = nn.Linear(1, dim_model, bias=False)
        self.cat_fc = nn.Linear(total_cat + 1, dim_model)
        self.cate_proj = nn.Sequential(
            nn.Linear(dim_model * 5, dim_model),
            nn.LayerNorm(dim_model),
        )

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex, in_cat, in_tg, in_dt, in_ts, first_block=True):
        device = in_ex.device

        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = F.one_hot(in_cat, num_classes=self.total_cat + 1).float()
            in_cat = self.cat_fc(in_cat)

            in_dt = in_dt.unsqueeze(-1)
            in_dt = self.dt_fc(in_dt)

            in_tg = self.embd_tg(in_tg)
            avg_in_tg_embed = in_tg.mean(dim=2)
            max_in_tg_embed = in_tg.max(dim=2).values

            # combining the embedings
            out = torch.cat([in_ex, in_cat, in_dt, avg_in_tg_embed, max_in_tg_embed], axis=2)
            out = self.cate_proj(out)
        else:
            out = in_ex

        in_pos = get_pos(in_ts)
        in_pos = self.embd_pos(in_pos)
        in_pos = self.pos_norm(in_pos)
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
    def __init__(self, dim_model, total_in, heads_de, seq_len, fc_out_dim=8):
        super().__init__()
        self.seq_len = seq_len - 1
        self.total_in = total_in
        self.embd_pos = nn.Embedding(self.seq_len, embedding_dim=dim_model)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.in_fc = nn.Linear(total_in, fc_out_dim, bias=False)
        self.el_fc = nn.Linear(1, fc_out_dim, bias=False)
        self.cate_proj = nn.Sequential(
            nn.Linear(fc_out_dim * 2, dim_model),
            nn.LayerNorm(dim_model),
        )

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, in_el, en_out, first_block=True):
        device = in_in.device

        if first_block:
            in_in = F.one_hot(in_in, num_classes=self.total_in).float()
            in_in = self.in_fc(in_in)

            in_el = in_el.unsqueeze(-1)
            in_el = self.el_fc(in_el)

            out = torch.cat([in_in, in_el], axis=2)
            out = self.cate_proj(out)
        else:
            out = in_in

        # in_pos = get_pos(self.seq_len, device)
        # in_pos = self.embd_pos(in_pos)
        # out = out + in_pos

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


def get_pos(task):
    # use sine positional embeddinds
    position_ids = (1 - (task <= task.roll(1, dims=1)).long()).cumsum(dim=1)
    return position_ids


class SAINT(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_tg, total_in,
                 heads_de, seq_len, num_fc_in_dim=2, num_fc_out_dim=32):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(Encoder_block(dim_model, heads_en, total_ex, total_cat, total_tg, seq_len), num_en)
        self.decoder = get_clones(Decoder_block(dim_model, total_in, heads_de, seq_len), num_de)
        # self.lstm = nn.LSTM(dim_model, num_fc_out_dim, 2)

        self.num_fc = nn.Linear(in_features=num_fc_in_dim, out_features=num_fc_out_dim)
        self.out_fc1 = nn.Linear(in_features=dim_model, out_features=num_fc_out_dim)
        self.out_fc2 = nn.Linear(in_features=num_fc_out_dim * 2, out_features=1)

    def forward(self, feat):
        in_ex = feat['in_ex']
        in_dt = feat['in_dt']
        in_el = feat['in_el']
        in_tg = feat['in_tag']
        in_cat = feat['in_cat']
        in_in = feat['in_de']
        in_ts = feat['in_ts']
        num_feat = feat['num_feat']

        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, in_tg, in_dt, in_ts, first_block=first_block)
            in_cat = in_ex

        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](in_in, in_el, en_out=in_ex, first_block=first_block)

        # in_in, _ = self.lstm(in_in)

        num_feat = self.num_fc(num_feat)
        in_in = self.out_fc1(in_in)
        in_in = torch.cat([in_in, num_feat], dim=2)
        in_in = self.out_fc2(in_in)

        return in_in.squeeze(-1)
