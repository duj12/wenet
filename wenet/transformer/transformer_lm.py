from typing import Any, List, Tuple, Dict, Union, Optional
from typeguard import check_argument_types
from pathlib import Path
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging, warnings

"""Positional Encoding Module."""
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(self._pre_hook)

    def _pre_hook(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        """Perform pre-hook in load_state_dict for backward compatibility.

        Note:
            We saved self.pe until v.0.5.2 but we have omitted it later.
            Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

        """
        k = prefix + "pe"
        if k in state_dict:
            state_dict.pop(k)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def rename_state_dict(
    old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]

class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args

def repeat(N, fn):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)])

class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in selfattention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length="11",
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        selfattention_layer_type="selfattn",
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        ctc_softmax=None,
        conditioning_layer_dim=None,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(self._pre_hook)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 8
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if selfattention_layer_type in [
            "selfattn",
            "rel_selfattn",
            "legacy_rel_selfattn",
        ]:
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = [
                (
                    attention_heads,
                    attention_dim,
                    attention_dropout_rate,
                )
            ] * num_blocks
        else:
            raise NotImplementedError(selfattention_layer_type)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, attention_dim
            )

    def _pre_hook(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
        rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
        # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
        rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8,),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    encoder_output = xs
                    # intermediate branches also require normalization.
                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)
                        xs = xs + self.conditioning_layer(intermediate_result)

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif pos_enc is None:

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.encoder = Encoder(
            idim=embed_unit,
            attention_dim=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="linear",
            pos_enc_class=pos_enc_class,
        )
        self.decoder = nn.Linear(att_unit, vocab_size)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(input)
        mask = self._target_mask(input)
        h, _ = self.encoder(x, mask)
        y = self.decoder(h)
        return y, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list



def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The difference from to_device() is changing to torch.Tensor if float or int
    value is found.

    The restriction to the returned value in DataParallel:
        The object must be
        - torch.cuda.Tensor
        - 1 or more dimension. 0-dimension-tensor sends warning.
        or a list, tuple, dict.

    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data

class ESPnetLanguageModel(nn.Module):
    def __init__(self, lm: nn.Module, vocab_size: int, ignore_id: int = 0):
        assert check_argument_types()
        super().__init__()
        self.lm = lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll) from transformer language model

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase

        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return {}

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # 1. Build LM model
        lm = TransformerLM(vocab_size=vocab_size, **args.lm_conf)
        espnet_lm = ESPnetLanguageModel(lm=lm, vocab_size=vocab_size, **args.model_conf)
        return espnet_lm

    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
    ):
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = cls.build_model(args)

        model.to(device)
        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            logging.info(f"load lm state from {model_file} for {device}.")
            model.load_state_dict(torch.load(model_file, map_location=device))

        return model, args
