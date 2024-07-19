import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import TimeXerEncoder, TimeXerEncoderLayer
from layers.SelfAttention_Family import FullAttention, TimeXerAttentionLayer
from layers.Embed import DataEmbedding_inverted,PatchEmbedding
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from typing import Callable, Optional

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # iTransformer
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.exo_i_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_i_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        padding = configs.stride
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Encoder-only architecture
        self.encoder = TimeXerEncoder(
            [
                TimeXerEncoderLayer(#计算embedding
                    TimeXerAttentionLayer(#计算self,cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)


    def forecast(self, seq_exogenous_x, seq_endogenous_x, seq_x_mark):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # exogenous_means = seq_exogenous_x.mean(1, keepdim=True).detach()
            # seq_exogenous_x = seq_exogenous_x - exogenous_means
            # exogenous_st = torch.sqrt(torch.var(seq_exogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # seq_exogenous_x /= exogenous_st

            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st


        _, _, N = seq_endogenous_x.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        exogenous_i_embedding = self.exo_i_embedding(seq_exogenous_x, seq_x_mark)  # covariates (e.g timestamp) can be also embedded as tokens
        endogenous_i_embedding = self.end_i_embedding(seq_endogenous_x, seq_x_mark)
        seq_endogenous_x_for_patch = seq_endogenous_x.permute(0, 2, 1)
        endogenous_patch_embedding, n_vars = self.patch_embedding(seq_endogenous_x_for_patch)

        self_attention_input = torch.concat([endogenous_patch_embedding,endogenous_i_embedding],dim=1)

        # B N E -> B N E
        encode_output, attns = self.encoder(self_attention_input, exogenous_i_embedding)

        # B N E -> B N S -> B S N
        dec_out = self.projector(encode_output).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, seq_exogenous_x, seq_endogenous_x, seq_x_mark):
        dec_out = self.forecast(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]