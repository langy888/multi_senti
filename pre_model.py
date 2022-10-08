# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

import math
import copy

from typing import Optional, List

import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter

import math

import torch
import torch.utils.checkpoint

from transformers.activations import gelu

from transformers.utils import logging

logger = logging.get_logger(__name__)



class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertIntermediate(nn.Module):
    def __init__(self, bert_dim):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(bert_dim, 4*bert_dim)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self, bert_dim):
        super(BertCoAttention, self).__init__()
        self.hidden_size = bert_dim
        self.num_attention_heads = bert_dim // 64
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, cl_att):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*12*75*49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # atention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        #contrastive attention
        if cl_att:
            attention_probs = 1 - attention_probs + s2_attention_mask 
            attention_probs = nn.Softmax(dim=-1)(attention_probs) 

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer b*12*75*64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        return context_layer


class BertOutput(nn.Module):
    def __init__(self, bert_dim):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(4*bert_dim, bert_dim)
        self.LayerNorm = BertLayerNorm(bert_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, bert_dim):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(bert_dim, bert_dim)
        self.LayerNorm = BertLayerNorm(bert_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self, bert_dim):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention(bert_dim)
        self.output = BertSelfOutput(bert_dim)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask, cl_att):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask, cl_att)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, bert_dim):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention(bert_dim)
        self.intermediate = BertIntermediate(bert_dim)
        self.output = BertOutput(bert_dim)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, cl_att):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask, cl_att)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self, bert_dim, layer_num=3):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(bert_dim)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, cl_att=0):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask, cl_att)
        return s1_hidden_states


class BertCSBlock(nn.Module):
    def __init__(self, bert_dim, concat_att=0, cross_coatt=0, self_coatt=0):
        super(BertCSBlock, self).__init__()
        self.cross_coatt = cross_coatt
        self.self_coatt = self_coatt
        self.concat_att = concat_att
        self.colayer_it = BertCrossAttentionLayer(bert_dim)
        self.colayer_ti = BertCrossAttentionLayer(bert_dim)        
        if concat_att:
            self.sf_att = BertCrossAttentionLayer(bert_dim)
        else:
            self.isf_att = BertCrossAttentionLayer(bert_dim)
            self.tsf_att = BertCrossAttentionLayer(bert_dim)

    def forward(self, text_f, image_f, text_m, image_m, text_image_m, cl_att=0):
        it_coatt = self.colayer_it(image_f, text_f, text_m, self.cross_coatt) #N Li H
        ti_coatt = self.colayer_ti(text_f, image_f, image_m, self.cross_coatt) #N Lt H
        if self.concat_att:
            text_image = torch.cat((ti_coatt, it_coatt), dim=1)
            text_image_output = self.sf_att(text_image, text_image, text_image_m, self.self_coatt)
            text_att = text_image_output[:, :-image_m.size(-1), :]
            image_att = text_image_output[:, -image_m.size(-1):, :]
        else:
            text_att = self.isf_att(ti_coatt, ti_coatt, text_m, self.self_coatt)
            image_att = self.tsf_att(it_coatt, it_coatt, image_m, self.self_coatt)
        return text_att, image_att


class BertCSEncoder(nn.Module):
    def __init__(self, bert_dim, concat_att=0, cross_coatt=0, self_coatt=0, layer_num=3):
        super(BertCSEncoder, self).__init__()
        layer = BertCSBlock(bert_dim, concat_att, cross_coatt, self_coatt)     
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, text_f, image_f, text_m, image_m, text_image_m, cl_att=0):
        for layer_module in self.layer:
            text_f, image_f = layer_module(text_f, image_f, text_m, image_m, text_image_m)
        return text_f, image_f

def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def init_detr_args(detr_path):
    import argparse
    from detr.models.detr import build
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    all_args = parser.parse_known_args()
    args = all_args[0]
    detr, detr_post = build(args)
    checkpoint = torch.load(detr_path, map_location='cpu')
    detr.load_state_dict(checkpoint['model'])
    return detr, detr_post

if __name__ == "__main__":
    model, _ = init_detr_args("/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/pretrained_model/detr-r50-e632da11.pth")
    print()