import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionSentence(nn.Conv2d):
    """ Position-wise Linear Layer for Sentence Block
    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvolutionSentence, self).__init__(
             in_channels, out_channels,
             kernel_size, stride, padding, dilation, groups, bias)

    def __call__(self, x):
        """Applies the linear layer.
        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).
        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).
        """
        x = x.unsqueeze(3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = torch.squeeze(y, 3)
        return y


class MultiHeadAttention(nn.Module):
    """ Multi Head Attention Layer for Sentence Blocks
    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.
    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True):
        super(MultiHeadAttention, self).__init__()

        if self_attention:
            self.W_QKV = ConvolutionSentence(
                n_units, n_units * 3, kernel_size=1, bias=False)
        else:
            self.W_Q = ConvolutionSentence(
                n_units, n_units, kernel_size=1, bias=False)
            self.W_KV = ConvolutionSentence(
                n_units, n_units * 2, kernel_size=1, bias=False)
        self.finishing_linear_layer = ConvolutionSentence(
            n_units, n_units, bias=False)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention

    def __call__(self, x, z=None, mask=None):
        # xp = self.xp
        h = self.h

        # temporary mask
        # mask = np.zeros((8, x.shape[2], x.shape[2]), dtype=np.bool)

        if self.is_self_attention:
            Q, K, V = torch.chunk(self.W_QKV(x), 3, axis=1)
        else:
            Q = self.W_Q(x)
            K, V = torch.chunk(self.W_KV(z), 2, axis=1)
        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency

        batch_Q = torch.cat(torch.chunk(Q, h, axis=1), axis=0)
        batch_K = torch.cat(torch.chunk(K, h, axis=1), axis=0)
        batch_V = torch.cat(torch.chunk(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))

        # mask = xp.concatenate([mask] * h, axis=0)
        batch_A = torch.matmul(batch_Q.permute(0, 2, 1), batch_K) * self.scale_score

        # Calculate Weighted Sum
        batch_A = batch_A.unsqueeze(1)
        batch_V = batch_V.unsqueeze(2)
        batch_C = torch.sum(batch_A * batch_V, axis=3)
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))

        C = torch.cat(torch.chunk(batch_C, h, axis=0), axis=1)
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units: int, ff_inner: int, ff_slope: float):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * ff_inner
        self.slope = ff_slope
        self.W_1 = ConvolutionSentence(n_units, n_inner_units)
        self.W_2 = ConvolutionSentence(n_inner_units, n_units)
        self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e, negative_slope=self.slope)
        e = self.W_2(e)
        return e


def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3
    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.
    """

    batch, units, length = x.shape
    e = x.permute(0, 2, 1).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.reshape((batch, length, out_units)).permute(0, 2, 1)
    assert(e.shape == (batch, out_units, length))
    return e


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNormalizationSentence(LayerNorm):
    """ Position-wise Linear Layer for Sentence Block
    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


class EncoderLayer(nn.Module):
    def __init__(self, n_units, ff_inner: int, ff_slope: float, h: int, dropout1: float, dropout2: float):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_units, h)
        self.feed_forward = FeedForwardLayer(n_units, ff_inner, ff_slope)
        self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

    def __call__(self, e, xx_mask):
        sub = self.self_attention(e, e, xx_mask)
        e = e + self.dropout1(sub)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + self.dropout2(sub)
        e = self.ln_2(e)
        return e


class TransformerV1(nn.Module):
    def __init__(self, dim_input: int, dim_enc: int, dim_fc: int, dim_output: int,
                 dim_cur_input: 4, ff_inner: int, ff_slope: float, head: int,
                 dropout1: float, dropout2: float, dropout3: float, **kwargs):
        '''
        from https://www.kaggle.com/toshik/37th-place-solution/notebook
        thank you for @toshi_k 's nice solution and imprements
        '''
        super(TransformerV1, self).__init__()

        self.dropout3 = nn.Dropout(dropout3)

        self.cur_fc1 = nn.Linear(dim_cur_input, 8)
        self.cur_fc2 = nn.Linear(8, 8)

        self.hist_conv1 = ConvolutionSentence(dim_input, int(dim_enc))
        self.hist_enc1 = EncoderLayer(int(dim_enc), ff_inner, ff_slope, head, dropout1, dropout2)
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(dim_input, head, 128, dropout=dropout1, activation='gelu'),
        #     2
        # )

        self.fc1 = nn.Linear(40, dim_fc)
        self.fc2 = nn.Linear(dim_fc, dim_output)

        self.sigmoid = nn.Sigmoid()

    def __call__(self, feats):

        out = self.predict(feats)
        return out

    def predict(self, feats, **kwargs):
        """
            query: [batch_size, feature]
            history: [batch_size, time_step, feature]
        """

        query = feats['current']
        history = feats['history']

        h_cur = F.leaky_relu(self.cur_fc1(query))
        h_cur = self.cur_fc2(h_cur)

        h_hist = history.permute(0, 2, 1)

        h_hist = self.hist_conv1(h_hist)
        h_hist = self.hist_enc1(h_hist, xx_mask=None)

        # h_hist = self.transformer(h_hist)

        h_hist_ave = torch.mean(h_hist, axis=2)
        h_hist_max, _ = torch.max(h_hist, axis=2)

        h = torch.cat([h_cur, h_hist_ave, h_hist_max], axis=1)

        h = self.dropout3(F.leaky_relu(self.fc1(h)))
        output = self.fc2(h)

        return output
