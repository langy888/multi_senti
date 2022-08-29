"""
Name: model
Date: 2022/4/11 上午10:25
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
#from torchvision.models import resnet50(pretrained=True)
import torch
import torch.nn.functional as F
import os
from transformers import BertConfig, BertForPreTraining, AutoTokenizer, AutoModel, CLIPConfig, CLIPModel, CLIPVisionModel,\
    ViTConfig, ViTModel, CLIPTextModel, AutoConfig
import math
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder, DynamicLSTM, GraphConvolution
import copy
from time import time
#os.environ['TORCH_HOME'] = '../../pretrained_model/input/' #setting the environment variable

class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, boxes=None, graph=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.boxes = boxes
        self.graph = graph

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, boxes=None, graph=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.boxes = boxes
        self.graph = graph


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


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class BertClassify(nn.Module):
    def __init__(self, opt, in_feature, dropout_rate=0.1):
        super(BertClassify, self).__init__()
        self.classify_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feature, 3),
            ActivateFun(opt)
        )

    def forward(self, inputs):
        return self.classify_linear(inputs)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = '' if not opt.debug else "/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/"

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert
            self.output_dim = self.model.encoder.layer[-1].output.dense.out_features
        elif 'roberta' in opt.text_model:
            #print("=================")
            #print(os.path.join(abl_path, 'pretrained_model', opt.text_model))
            self.config = AutoConfig.from_pretrained(os.path.join(abl_path, 'pretrained_model', opt.text_model))
            self.model = AutoModel.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.text_model))
            self.output_dim = self.model.encoder.layer[-1].output.dense.out_features
        elif 'clip' in opt.text_model:
            self.config = AutoConfig.from_pretrained(os.path.join(abl_path, 'pretrained_model', opt.text_model))
            clipmodel = CLIPModel.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.text_model))
            self.model = clipmodel.text_model
            self.output_dim = clipmodel.config.text_config.hidden_size
            #print("GPU init", clipmodel.config.text_config.max_position_embeddings)
            del clipmodel


        for param in self.model.parameters():
            param.requires_grad = True

        #self.output_dim = self.model.encoder.layer[-1].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output


class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        abl_path = '' if not opt.debug else "/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/"
        self.image_model = opt.image_model
        if 'resnet' in opt.image_model:
            if opt.image_model == 'resnet-152':
                self.resnet = cv_models.resnet152(pretrained=True)
            elif opt.image_model == 'resnet-101':
                self.resnet = cv_models.resnet101(pretrained=True)
            elif opt.image_model == 'resnet-50':
                self.resnet = cv_models.resnet50(pretrained=True)
            elif opt.image_model == 'resnet-34':
                self.resnet = cv_models.resnet34(pretrained=True)
            elif opt.image_model == 'resnet-18':
                self.resnet = cv_models.resnet18(pretrained=True)
            self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
            self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
            self.output_dim = self.resnet_encoder[7][2].conv3.out_channels
            for param in self.resnet.parameters():
                if opt.fixed_image_model:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif 'vit' in opt.image_model:
            #self.feature_extractor = ViTFeatureExtractor.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.image_model))
            self.vit = ViTModel.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.image_model)) 
            self.output_dim = self.vit.config.hidden_size

            for param in self.vit.parameters():
                if opt.fixed_image_model:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif 'clip' in opt.image_model:
            #self.feature_extractor = ViTFeatureExtractor.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.image_model))
            clipmodel = CLIPModel.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.image_model)) 
            self.clip = clipmodel.vision_model
            self.output_dim = clipmodel.config.vision_config.hidden_size
            del clipmodel

            for param in self.clip.parameters():
                if opt.fixed_image_model:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        if 'resnet' in self.image_model:
            image_encoder = self.resnet_encoder(images)
            # image_encoder = self.conv_output(image_encoder)
            image_cls = self.resnet_avgpool(image_encoder)
            image_cls = torch.flatten(image_cls, 1)
            return image_encoder, image_cls
        if 'vit' in  self.image_model:
            outputs = self.vit(pixel_values=images)
            image_encoder = outputs.last_hidden_state
            image_cls = outputs.pooler_output
            return image_encoder, image_cls
        if 'clip' in  self.image_model:
            outputs = self.clip(pixel_values=images)
            image_encoder = outputs.last_hidden_state
            image_cls = outputs.pooler_output
            return image_encoder, image_cls


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.opt = opt
        self.gcn = opt.gcn
        if not self.gcn:
            if 'clip' in opt.image_model and opt.use_clip_proj:
                clipmodel = CLIPModel.from_pretrained()
            self.fuse_type = opt.fuse_type
            self.image_output_type = opt.image_output_type
            self.zoom_value = math.sqrt(opt.tran_dim)
            self.save_image_index = 0
            self.no_extra_img_trans = opt.no_extra_img_trans

            self.text_model = TextModel(opt)
            self.image_model = ImageModel(opt)
            if 'clip' in opt.text_model:
                self.text_config = copy.deepcopy(BertConfig.from_pretrained(opt.abl_path+'bert_base'))
                self.image_config = copy.deepcopy(BertConfig.from_pretrained(opt.abl_path+'bert_base'))
            else:
                self.text_config = copy.deepcopy(self.text_model.get_config())
                self.image_config = copy.deepcopy(self.text_model.get_config())

            self.text_config.num_attention_heads = opt.tran_dim // 64
            self.text_config.hidden_size = opt.tran_dim
            self.text_config.num_hidden_layers = opt.tran_num_layers

            self.image_config.num_attention_heads = opt.tran_dim // 64
            self.image_config.hidden_size = opt.tran_dim
            self.image_config.num_hidden_layers = opt.image_num_layers

            if self.text_config.is_decoder:
                self.use_cache = self.text_config.use_cache
            else:
                self.use_cache = False

            self.text_image_encoder = RobertaEncoder(self.text_config)
            self.image_encoder = RobertaEncoder(self.image_config) \
                if not self.no_extra_img_trans else None

            self.text_change = nn.Sequential(
                nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
                ActivateFun(opt)
            )
            self.image_change = nn.Sequential(
                nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
                ActivateFun(opt)
            )
            self.image_cls_change = nn.Sequential(
                nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
                ActivateFun(opt)
            )

            if 'clip' in opt.image_model and opt.use_clip_proj:
                
                self.text_change = nn.Sequential(
                    clipmodel.text_projection,
                    ActivateFun(opt)
                )
                self.image_change = nn.Sequential(
                    clipmodel.visual_projection,
                    ActivateFun(opt)
                )
                self.image_cls_change = nn.Sequential(
                    clipmodel.visual_projection,
                    ActivateFun(opt)
                )
                del clipmodel

            # self.transformer_embedding_layernorm = nn.Sequential(
            #     nn.LayerNorm(opt.tran_dim),
            #     nn.Dropout(opt.l_dropout)
            # )

            #self.image_encoder = None

            if self.fuse_type == 'att':
                self.output_attention = nn.Sequential(
                    nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                    ActivateFun(opt),
                    nn.Linear(opt.tran_dim // 2, 1)
                )

            self.output_classify = nn.Sequential(
                nn.Dropout(opt.l_dropout),
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 3)
            )
        else:
            self.fuse_type = opt.fuse_type
            self.image_output_type = opt.image_output_type
            self.zoom_value = math.sqrt(opt.tran_dim)
            self.save_image_index = 0
            self.no_extra_img_trans = opt.no_extra_img_trans

            self.text_model = TextModel(opt)
            self.image_model = ImageModel(opt)
            self.text_config = copy.deepcopy(self.text_model.get_config())
            self.image_config = copy.deepcopy(self.text_model.get_config())


            # self.text_config.num_attention_heads = opt.tran_dim // 64
            # self.text_config.hidden_size = opt.tran_dim
            # self.text_config.num_hidden_layers = opt.tran_num_layers

            # self.image_config.num_attention_heads = opt.tran_dim // 64
            # self.image_config.hidden_size = opt.tran_dim
            # self.image_config.num_hidden_layers = opt.image_num_layers
            # self.use_cache = False
            
            self.gc1 = GraphConvolution(2*512, 2*512)
            self.gc2 = GraphConvolution(2*512, 2*512)
            self.vit_fc = nn.Linear(self.image_model.get_output_dim(), 2*512)
            self.text_lstm = DynamicLSTM(opt.tran_dim, 512, num_layers=1, batch_first=True, bidirectional=True)
            #self.output_classify = nn.Linear(2*512, 2)
            self.output_classify = nn.Sequential(
                nn.Dropout(opt.l_dropout),
                nn.Linear(1024, 1024 // 2),
                ActivateFun(opt),
                nn.Linear(1024 // 2, 3)
            )  

    def DropEdge(self, b_graph):
        rand_mask = torch.rand(b_graph.shape)
        zeros = torch.zeros_like(b_graph)
        if self.opt.cuda:
            rand_mask = rand_mask.cuda()
            zeros = zeros.cuda()
        res = torch.where(rand_mask >= 0.2, b_graph, zeros)
        return res

    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask, boxes=None, graph=None):
        if not self.gcn:
            text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
            text_cls = text_encoder.pooler_output # N * H
            text_cls_init = self.text_change(text_cls)
            text_encoder = text_encoder.last_hidden_state # N * L * H
            text_init = self.text_change(text_encoder) # N * L * 768
            image_encoder, image_cls = self.image_model(image_inputs) # N * X(HW) * Hi  N * Hi 
            if self.image_output_type == 'all':
                if len(image_encoder.shape) == 4:
                    image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))  # N * X(HW) * Hi 
                image_encoder_init = self.image_change(image_encoder) # N * Li * 768（train dim） 
                image_cls_init = self.image_cls_change(image_cls) # N * 768（train dim） 
                image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1) #N * （Li + 1）* 768
            else:
                image_cls_init = self.image_cls_change(image_cls)
                image_init = image_cls_init.unsqueeze(1)

            image_mask = text_image_mask[:, -image_init.size(1):]
            extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())

            if self.image_encoder:
                image_init = self.image_encoder(image_init,
                                                    attention_mask=None,
                                                    head_mask=None,
                                                    encoder_hidden_states=None,
                                                    encoder_attention_mask=extended_attention_mask,
                                                    past_key_values=None,
                                                    use_cache=self.use_cache,
                                                    output_attentions=self.text_config.output_attentions,
                                                    output_hidden_states=(self.text_config.output_hidden_states),
                                                    return_dict=self.text_config.use_return_dict
                                                    )
                image_init = image_init.last_hidden_state

            text_image_cat = torch.cat((text_init, image_init), dim=1) #  N * (L + Li + 1) * 768

            extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_inputs.size())
            text_image_transformer = self.text_image_encoder(text_image_cat,
                                                    attention_mask=extended_attention_mask,
                                                    head_mask=None,
                                                    encoder_hidden_states=None,
                                                    encoder_attention_mask=extended_attention_mask,
                                                    past_key_values=None,
                                                    use_cache=self.use_cache,
                                                    output_attentions=self.text_config.output_attentions,
                                                    output_hidden_states=(self.text_config.output_hidden_states),
                                                    return_dict=self.text_config.use_return_dict)
            text_image_transformer = text_image_transformer.last_hidden_state
            #text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()

            if self.fuse_type == 'max':
                text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()
                text_image_output = torch.max(text_image_transformer, dim=2)[0]
            elif self.fuse_type == 'att':
                #text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

                #text_image_mask = text_image_mask.permute(1, 0).contiguous()
                #text_image_mask = text_image_mask[0:text_image_output.size(1)]
                #text_image_mask = text_image_mask.permute(1, 0).contiguous()

                text_image_alpha = self.output_attention(text_image_transformer)
                text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
                text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

                # text_image_alpha_sum = torch.sum(text_image_alpha, dim=0)/text_image_alpha.shape[0]
                # image_alpha = text_image_alpha_sum[-50:]
                # image_alpha_sum = torch.sum(image_alpha)
                # print("IMAGE ATT:", image_alpha_sum)

                text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_transformer).sum(dim=1)
            elif self.fuse_type == 'ave':
                text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()
                text_image_length = text_image_transformer.size(2)
                text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
            else:
                raise Exception('fuse_type设定错误')
            return text_image_output, text_cls_init, image_cls_init
        else:
            text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
            text_cls = text_encoder.pooler_output # N * H
            #text_cls_init = self.text_change(text_cls)
            text_encoder = text_encoder.last_hidden_state # N * L * H
            #text_init = self.text_change(text_encoder) # N * L * 768

            image_encoder, image_cls = self.image_model(image_inputs) # N * X(HW) * Hi  N * Hi 

            box_fs = []
            for  box in boxes:
                _, box_cls = self.image_model(box)
                box_fs.append(box_cls)

            box_fs = torch.stack(box_fs) # N x 10 x outdim
            box_vit = torch.cat((box_fs, image_cls.unsqueeze(1)), dim = 1)


            bert_text_len = torch.sum(text_inputs != 0, dim=-1)
            text_out, (_, _) = self.text_lstm(text_encoder, bert_text_len)


            box_vit = self.vit_fc(box_vit)
            features = torch.cat([text_out, box_vit],dim = 1)

            if self.training:
                graph = self.DropEdge(graph)
            x = F.relu(self.gc1(features, graph))
            x_ensemble = F.relu(self.gc2(x, graph)) 


            alpha_mat = torch.matmul(features, x_ensemble.transpose(1, 2))
            alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
            x_ensemble = torch.matmul(alpha, x_ensemble).squeeze(1)

            output = self.output_classify(x_ensemble)            
            return output, x_ensemble, None

class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        self.temperature = opt.temperature
        self.set_cuda = opt.cuda
        self.gcn = opt.gcn
        self.clloss = 1
        self.cllabel = 1
        self.claug = 0

        if not self.gcn:
            self.orgin_linear_change = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim, opt.tran_dim)
            )

            self.augment_linear_change = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim, opt.tran_dim)
            )

            self.output_classify = nn.Sequential(
                nn.Dropout(opt.l_dropout),
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 2)
            )
        else:
            self.orgin_linear_change = nn.Sequential(
                nn.Linear(1024, 1024),
                ActivateFun(opt),
                nn.Linear(1024, 1024)
            )

            self.augment_linear_change = nn.Sequential(
                nn.Linear(1024, 1024),
                ActivateFun(opt),
                nn.Linear(1024, 1024)
            )

            self.output_classify = nn.Sequential(
                nn.Dropout(opt.l_dropout),
                nn.Linear(1024, 1024 // 2),
                ActivateFun(opt),
                nn.Linear(1024 // 2, 3)
            )           

    def forward(self, data_orgin: ModelParam, data_augment: ModelParam = None, labels=None, target_labels=None, text_sep_labels=None, image_sep_labels=None):
        if not self.gcn:
            orgin_res, orgin_text_cls, orgin_image_cls = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                        data_orgin.images, data_orgin.text_image_mask)
            output = self.output_classify(orgin_res) # N * 3 分类结果

        else:
            output, orgin_res, orgin_image_cls = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                        data_orgin.images, data_orgin.text_image_mask, 
                                                                        data_orgin.boxes, data_orgin.graph)
            if data_augment:
                outputaug, augment_res, aug_image_cls = self.fuse_model(data_augment.texts, data_augment.bert_attention_mask,
                                                                            data_augment.images, data_augment.text_image_mask,
                                                                            data_augment.boxes, data_augment.graph)

        if data_augment and not self.gcn:
            augment_res, augment_text_cls, augment_image_cls = self.fuse_model(data_augment.texts, data_augment.bert_attention_mask,
                                                                               data_augment.images, data_augment.text_image_mask)
            if self.clloss:
                orgin_res_change = self.orgin_linear_change(orgin_res) # N * 768
                if self.claug:
                    orgin_res_change = self.orgin_linear_change(orgin_res) # N * 768
                    augment_res_change = self.augment_linear_change(augment_res) # N * 768

                    l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T]) # N * N 原始和增强后结果的相似度
                    cl_lables = torch.arange(l_pos_neg.size(0)) # arange（N）
                    if self.set_cuda:
                        cl_lables = cl_lables.cuda()
                    l_pos_neg /= self.temperature
                else:
                    l_pos_neg, cl_lables = None, None

                if self.cllabel:
                    l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T]) # N * N 原始结果之间的相似度
                    l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
                    l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN

                    cl_self_labels = target_labels[labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
                    for index in range(1, orgin_res.size(0)):
                        cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

                    l_pos_neg_self = l_pos_neg_self / self.temperature
                    cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
                    cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)
                    
                else:
                    cl_self_loss = None
                
            else:
                l_pos_neg, cl_lables, cl_self_loss = None, None, None


            other_loss = []
            # if text_sep_labels:
            # #### TEXT ####
            #     l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_text_cls, orgin_text_cls.T]) # N * N 原始结果之间的相似度
            #     l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
            #     l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN
            #     text_labels = text_sep_labels[-1]
            #     cl_self_labels = text_sep_labels[text_labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
            #     for index in range(1, text_labels.size(0)):
            #         cl_self_labels = torch.cat((cl_self_labels, text_sep_labels[text_labels[index]] + index*text_labels.size(0)), 0)

            #     l_pos_neg_self = l_pos_neg_self / self.temperature
            #     cl_self_text_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            #     cl_self_text_loss = - cl_self_text_loss.sum() / cl_self_labels.size(0)
            #     other_loss.append(cl_self_text_loss)

            #     #### IMAGE ####
            #     l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_image_cls, orgin_image_cls.T]) # N * N 原始结果之间的相似度
            #     l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
            #     l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN
            #     image_labels = image_sep_labels[-1]
            #     cl_self_labels = image_sep_labels[image_labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
            #     for index in range(1, image_labels.size(0)):
            #         cl_self_labels = torch.cat((cl_self_labels, image_sep_labels[image_labels[index]] + index*image_labels.size(0)), 0)

            #     l_pos_neg_self = l_pos_neg_self / self.temperature
            #     cl_self_image_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            #     cl_self_image_loss = - cl_self_image_loss.sum() / cl_self_labels.size(0)
            #     other_loss.append(cl_self_image_loss)

            #     #### AUG TEXT ####
            #     l_pos_neg_self = torch.einsum('nc,ck->nk', [augment_text_cls, augment_text_cls.T]) # N * N 原始结果之间的相似度
            #     l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
            #     l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN
            #     text_labels = text_sep_labels[-1]
            #     cl_self_labels = text_sep_labels[text_labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
            #     for index in range(1, text_labels.size(0)):
            #         cl_self_labels = torch.cat((cl_self_labels, text_sep_labels[text_labels[index]] + index*text_labels.size(0)), 0)

            #     l_pos_neg_self = l_pos_neg_self / self.temperature
            #     cl_aug_text_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            #     cl_aug_text_loss = - cl_aug_text_loss.sum() / cl_self_labels.size(0)
            #     other_loss.append(cl_aug_text_loss)

            #     #### AUG IMAGE ####
            #     l_pos_neg_self = torch.einsum('nc,ck->nk', [augment_image_cls, augment_image_cls.T]) # N * N 原始结果之间的相似度
            #     l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
            #     l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN
            #     image_labels = image_sep_labels[-1]
            #     cl_self_labels = image_sep_labels[image_labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
            #     for index in range(1, image_labels.size(0)):
            #         cl_self_labels = torch.cat((cl_self_labels, image_sep_labels[image_labels[index]] + index*image_labels.size(0)), 0)

            #     l_pos_neg_self = l_pos_neg_self / self.temperature
            #     cl_aug_image_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            #     cl_aug_image_loss = - cl_aug_image_loss.sum() / cl_self_labels.size(0)
            #     other_loss.append(cl_aug_image_loss)

            return output, l_pos_neg, cl_lables, cl_self_loss, other_loss\
                #(cl_self_text_loss, cl_self_image_loss, cl_aug_text_loss, cl_aug_image_loss)
        elif self.gcn and self.training:
            
            orgin_res_change = self.orgin_linear_change(orgin_res)

            l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T]) # N * N 原始结果之间的相似度
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1) # log（ softmax（ input ））
            l_pos_neg_self = l_pos_neg_self.view(-1) # 展平为 1， NN

            cl_self_labels = target_labels[labels[0]] # 对于每个label 计算他和其他相同label样本间的相似度和
            for index in range(1, orgin_res.size(0)):
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

            if data_augment:
                augment_res_change = self.augment_linear_change(augment_res) # N * 768

                l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T]) # N * N 原始和增强后结果的相似度
                cl_lables = torch.arange(l_pos_neg.size(0)) # arange（N）
                if self.set_cuda:
                    cl_lables = cl_lables.cuda()
                l_pos_neg /= self.temperature

            return output, (cl_self_loss, l_pos_neg, cl_lables)
        else:
            return output


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = CLModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
