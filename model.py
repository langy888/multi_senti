"""
Name: model
Date: 2022/4/11 上午10:25
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, AutoTokenizer, AutoModel,\
    ViTConfig, ViTModel, ViTFeatureExtractor
import math
import matplotlib.pyplot as plt
from pre_model import BertCrossEncoder
import copy
#os.environ['TORCH_HOME'] = '../../pretrained_model/input/' #setting the environment variable


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, emoji=None, hashtag=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.emoji = emoji
        self.hashtag = hashtag

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, emoji=None, hashtag=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.emoji = emoji
        self.hashtag = hashtag


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
        else:
            #print("=================")
            #print(os.path.join(abl_path, 'pretrained_model', opt.text_model))
            self.config = BertConfig.from_pretrained(os.path.join('pretrained_model', opt.text_model))
            self.model = AutoModel.from_pretrained(os.path.join('pretrained_model',  opt.text_model))
            #self.model = self.model.roberta

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

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
            self.vit = ViTModel.from_pretrained(os.path.join(abl_path, 'pretrained_model',  opt.image_model)) 
            self.output_dim = self.vit.config.hidden_size

            for param in self.vit.parameters():
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
            #pixel_values = self.feature_extractor(images=images, return_tensors='pt')
            outputs = self.vit(pixel_values = images)
            image_encoder = outputs.last_hidden_state
            image_cls = outputs.pooler_output
            return image_encoder, image_cls


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)

        self.text_image_encoder = BertCrossEncoder(opt.tran_dim, 5)
        self.image_encoder = BertCrossEncoder(opt.tran_dim, 1)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.text_cls_change = nn.Sequential(
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

        self.ftext_cls_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.fimage_cls_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.linear_output_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim()+self.image_model.get_output_dim()+opt.tran_dim, opt.tran_dim),
            ActivateFun(opt)
        )

        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)

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

    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask, emoji, hashtag):
        #text_inputs = torch.cat((text_inputs, hashtag, emoji), dim=1)
        #text_inputs = torch.cat((text_inputs, emoji), dim=1)
        assert text_inputs.size(0) == bert_attention_mask.size(0)
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        text_cls = text_encoder.pooler_output # N * H
        text_cls_init = self.text_cls_change(text_cls)
        text_encoder = text_encoder.last_hidden_state # N * L * H
        text_init = self.text_change(text_encoder) # N * L * H
        image_encoder, image_cls = self.image_model(image_inputs) # N * X * Hi  N * Hi 

        if self.image_output_type == 'all':
            image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))
            image_encoder_init = self.image_change(image_encoder)
            image_cls_init = self.image_cls_change(image_cls)
            image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)
        else:
            image_cls_init = self.image_cls_change(image_cls)
            image_init = image_cls_init.unsqueeze(1)

        image_mask = text_image_mask[:, -image_init.size(1):]
        extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())

        image_init = self.image_encoder(image_init, image_init, extended_attention_mask)

        ####非crossatt删掉
        text_mask =  text_image_mask[:, :-image_init.size(1)]
        text_extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_mask, text_inputs.size())

        text_image_cat = torch.cat((text_init, image_init), dim=1)

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_inputs.size())
        text_image_output = self.text_image_encoder(text_image_cat, text_image_cat, extended_attention_mask)

        text_image_clatt = self.text_image_encoder(image_init, text_init, text_extended_attention_mask, cl_att=1) # N, 50, 768

        fused_text_cls = text_image_output[:,0,:]
        fused_img_cls = text_image_output[:,-50,:]

        fused_text_cls = self.ftext_cls_change(fused_text_cls)
        fused_img_cls = self.fimage_cls_change(fused_img_cls)

        text_image_output = text_image_output[:, :-image_init.size(1)]

        #text_image_mask = torch.cat((text_image_mask,image_mask), dim=1)
        text_image_output = torch.cat((text_image_output,text_image_clatt), dim=1)

        text_image_alpha = self.output_attention(text_image_output)
        text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
        text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)

        return text_image_output, text_cls_init, image_cls_init, fused_text_cls, fused_img_cls


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        self.temperature = opt.temperature
        self.set_cuda = opt.cuda
        self.clloss = 1
        self.cll = opt.cll
        self.it_cl = opt.it
        self.cla = opt.cla
        self.tt_cla = opt.tt
        self.ii_cla = opt.ii
        self.ff_cl = opt.ff
        #self.mmt
        self.critertion = nn.CrossEntropyLoss()

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
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, data_orgin: ModelParam, data_augment: ModelParam = None, labels=None, target_labels=None):
        orgin_res, orgin_text_cls, orgin_image_cls, ftext, fimage = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask,
                                                                     data_orgin.emoji, data_orgin.hashtag)
        output = self.output_classify(orgin_res)

        if data_augment:
            augment_res, augment_text_cls, augment_image_cls,_ ,_ = self.fuse_model(data_augment.texts, data_augment.bert_attention_mask,
                                                                            data_augment.images, data_augment.text_image_mask,
                                                                            data_augment.emoji, data_augment.hashtag)
            augment_res_change = self.augment_linear_change(augment_res)
            orgin_res_change = self.orgin_linear_change(orgin_res)

            cla_loss = 0
            if self.cla:
                #l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T])
                l_pos_neg = torch.mm(orgin_res_change, augment_res_change.T)
                cl_lables = torch.arange(l_pos_neg.size(0))
                if self.set_cuda:
                    cl_lables = cl_lables.cuda()
                l_pos_neg /= self.temperature
                cla_loss = self.critertion(l_pos_neg, cl_lables)

            cl_self_loss = 0
            if self.cll:     
                #l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T])
                l_pos_neg_self = torch.mm(orgin_res_change, orgin_res_change.T)
                l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
                l_pos_neg_self = l_pos_neg_self.view(-1)

                cl_self_labels = target_labels[labels[0]]
                for index in range(1, orgin_res.size(0)):
                    cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

                l_pos_neg_self = l_pos_neg_self / self.temperature
                cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
                cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

            it_loss = 0
            if self.it_cl:
                #it_pos_neg = torch.einsum('nc,ck->nk', [orgin_text_cls, orgin_image_cls.T])
                it_pos_neg = torch.mm(orgin_text_cls, orgin_image_cls.T)
                it_cl_lables = torch.arange(it_pos_neg.size(0))
                if self.set_cuda:
                    it_cl_lables = it_cl_lables.cuda()
                it_pos_neg /= self.temperature    
                it_loss = self.critertion(it_pos_neg, it_cl_lables)           

            tt_loss = 0           
            if self.tt_cla:
                #tt_pos_neg = torch.einsum('nc,ck->nk', [orgin_text_cls, augment_text_cls.T])
                tt_pos_neg = torch.mm(orgin_text_cls, augment_text_cls.T)
                tt_cl_lables = torch.arange(tt_pos_neg.size(0))
                if self.set_cuda:
                    tt_cl_lables = tt_cl_lables.cuda()
                tt_pos_neg /= self.temperature 
                tt_loss = self.critertion(tt_pos_neg, tt_cl_lables)                  

            ii_loss = 0               
            if self.ii_cla:
                #ii_pos_neg = torch.einsum('nc,ck->nk', [orgin_image_cls, augment_image_cls.T])
                ii_pos_neg = torch.mm(orgin_image_cls, augment_image_cls.T)
                ii_cl_lables = torch.arange(ii_pos_neg.size(0))
                if self.set_cuda:
                    ii_cl_lables = ii_cl_lables.cuda()
                ii_pos_neg /= self.temperature   
                ii_loss = self.critertion(ii_pos_neg, ii_cl_lables) 

            ff_loss = 0
            if self.ff_cl:
                fit_pos_neg = torch.mm(ftext, fimage.T) #orgin_text_cls, orgin_image_cls
                fit_cl_lables = torch.arange(fit_pos_neg.size(0))
                if self.set_cuda:
                    fit_cl_lables = fit_cl_lables.cuda()
                fit_pos_neg /= self.temperature   
                ff_loss = self.critertion(fit_pos_neg, fit_cl_lables)

            # if self.fo_cl:
            #     tt_pos_neg = torch.mm(ftext, orgin_image_cls.T)
            #     tt_cl_lables = torch.arange(tt_pos_neg.size(0))
            #     if self.set_cuda:
            #         tt_cl_lables = tt_cl_lables.cuda()
            #     tt_pos_neg /= self.temperature 
            #     tt_loss = self.critertion(tt_pos_neg, tt_cl_lables)          

            #     ii_pos_neg = torch.mm(fimage, orgin_text_cls.T)
            #     ii_cl_lables = torch.arange(ii_pos_neg.size(0))
            #     if self.set_cuda:
            #         ii_cl_lables = ii_cl_lables.cuda()
            #     ii_pos_neg /= self.temperature   
            #     ii_loss = self.critertion(ii_pos_neg, ii_cl_lables) 


            return output, cl_self_loss, cla_loss, (it_loss, tt_loss, ii_loss, ff_loss)
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
