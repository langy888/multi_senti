"""
Name: data_process
Date: 2022/4/11 上午10:25
Version: 1.0
"""

from collections import defaultdict
from PIL import Image
from PIL import ImageFile
from PIL import TiffImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import json
import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from util.image_augmentation.augmentations import RandAugment
import copy
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from transformers import ViTFeatureExtractor
import pickle
from time import time

class SentenceDataset(Dataset):
    def __init__(self, opt, data_path, text_tokenizer, photo_path, image_transforms, data_type, data_translation_path=None, image_coordinate=None, text_image_labels_path=None, gcn_root=None):
        self.gcn = opt.gcn
        self.data_type = data_type
        self.dataset_type = opt.data_type
        self.photo_path = photo_path
        self.image_transforms = image_transforms

        file_read = open(data_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()

        if text_image_labels_path:
            with open(text_image_labels_path,'r') as f:
                self.text_image_label = json.load(f)
        else:
            self.text_image_label = None

        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        self.text_to_id = []
        if opt.gcn:
            self.aug_text_to_id = []

        for data in file_content:
            self.data_id_list.append(data['id'])
            self.text_list.append(data['text'])
            self.label_list.append(data['emotion_label'])

        if self.dataset_type != 'meme7k':
            self.image_id_list = [str(data_id) + '.jpg' for data_id in self.data_id_list]
        else:
            self.image_id_list = self.data_id_list

        file_read = open(data_translation_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()
        self.data_translation_id_to_text_dict = {data['id']: data['text_translation'] for data in file_content}

        if opt.text_model == 'bert-base':
            pass

        if self.gcn:
            #self.image_id_box = defaultdict(list)
            gcn_data = pickle.load(open(gcn_root,'rb'))
            self.gcn_data = gcn_data
            for _, id_ in enumerate(self.data_id_list):
                value = gcn_data[id_] 
                tokens = value['tokens']
                self.text_to_id.append(text_tokenizer.convert_tokens_to_ids(tokens))

                aug_tokens = value['aug_tokens']
                self.aug_text_to_id.append(text_tokenizer.convert_tokens_to_ids(aug_tokens))

        else:
            self.text_token_list = [text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in tqdm(self.text_list, desc='convert text to token')]
            self.text_translation_id_to_token_list = {index: text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for index, text in self.data_translation_id_to_text_dict.items()}
            self.text_token_list = [text if len(text) < opt.word_length else text[0: opt.word_length] for text in
                                    self.text_token_list] #截取过长词
            self.text_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in
                            tqdm(self.text_token_list, desc='convert text to id')]
            self.text_translation_id_to_token_list = {index: text_token if len(text_token) < opt.word_length else text_token[0:opt.word_length] for index, text_token in
                                                    self.text_translation_id_to_token_list.items()}
            self.text_translation_to_id = {index: text_tokenizer.convert_tokens_to_ids(text_token) for index, text_token in self.text_translation_id_to_token_list.items()}

        if 'clip' in opt.text_model:
            new_text_to_id = []
            for text_ids in self.text_to_id:
                if len(text_ids) > 77:
                    text_ids = text_ids[:76] + [text_ids[-1]]
                new_text_to_id.append(text_ids)
            self.text_to_id = new_text_to_id
            new_text_translation_to_id = {}
            for index, text_ids in self.text_translation_to_id.items():
                if len(text_ids) > 77:
                    text_ids = text_ids[:76] + [text_ids[-1]]
                new_text_translation_to_id[index] = text_ids
            self.text_translation_to_id = new_text_translation_to_id

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        id_ = self.data_id_list[index]
        image_path = self.photo_path + '/' + str(id_) + '.jpg'
        image_read = Image.open(image_path)
        image_read.load()

        text_sep_label, image_sep_label = None , None
        if self.text_image_label:
            text_sep_label, image_sep_label = self.text_image_label[str(id_)].split(",")
            text_sep_label = int(text_sep_label)
            image_sep_label = int(image_sep_label)


        if self.gcn:
            value = self.gcn_data[id_]
            graph = value['graph'] 
            image_graph = value["image_graph"]
            boxes = value["box"]
            aug_graph = value['aug_graph']
            aug_image_graph = value['aug_image_graph']
            boxes_images = [image_read.crop(box) for box in boxes]
            box_tensors = [self.image_transforms(box) for box in boxes_images]
            image_origin = self.image_transforms(image_read)
            # image_augment = image_origin
            # if self.data_type == 1:
            #     image_augment = copy.deepcopy(image_read)
            #     image_augment = self.image_transforms(image_augment)
            return self.text_to_id[index], image_origin, self.label_list[index], box_tensors, graph, image_graph,\
                id_, self.aug_text_to_id[index], aug_graph, aug_image_graph
        else:
            image_origin = self.image_transforms(image_read)
            image_augment = image_origin
            if self.data_type == 1:
                image_augment = copy.deepcopy(image_read)
                image_augment = self.image_transforms(image_augment)
            return self.text_to_id[index], image_origin, self.label_list[index], self.text_translation_to_id[id_], \
                image_augment, self.data_id_list[index], text_sep_label, image_sep_label
        


class Collate():
    def __init__(self, opt):
        self.gcn = opt.gcn
        self.text_length_dynamic = opt.text_length_dynamic
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.word_length

        self.image_mask_num = 0
        if opt.image_output_type == 'cls':
            self.image_mask_num = 1
        elif opt.image_output_type == 'all':
            if 'resnet' in opt.image_model:
                self.image_mask_num = 50
            elif 'vit' in opt.image_model:
                self.image_mask_num = 198

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        if not self.gcn:
            text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
            image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
            label = torch.LongTensor([b[2] for b in batch_data])
            text_translation_to_id = [torch.LongTensor(b[3]) for b in batch_data]
            image_augment = torch.FloatTensor([np.array(b[4]) for b in batch_data])
            origin_indexes = [b[5] for b in batch_data]
            text_labels = torch.LongTensor([b[6] for b in batch_data if b[6]])
            image_labels = torch.LongTensor([b[7] for b in batch_data if b[6]])

            data_length = [text.size(0) for text in text_to_id]
            data_translation_length = torch.LongTensor([text.size(0) for text in text_translation_to_id])

            max_length = max(data_length)
            if max_length < self.min_length:
                # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
                text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0))))) #padding 往后pad 0 到min_length长度
                max_length = self.min_length

            max_translation_length = max(data_translation_length)
            if max_translation_length < self.min_length:
                # 这个地方随便选一个只要保证翻译的文本里面某一个大于设定的min_length就可以保证后续不会报错了
                text_translation_to_id[0] = torch.cat((text_translation_to_id[0], torch.LongTensor([0] * (self.min_length - text_translation_to_id[0].size(0)))))
                max_translation_length = self.min_length

            text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
            text_translation_to_id = run_utils.pad_sequence(text_translation_to_id, batch_first=True, padding_value=0)

            bert_attention_mask = []
            text_image_mask = []
            for length in data_length:
                text_mask_cell = [1] * length
                text_mask_cell.extend([0] * (max_length - length))
                bert_attention_mask.append(text_mask_cell[:])

                text_mask_cell.extend([1] * self.image_mask_num)
                text_image_mask.append(text_mask_cell[:])

            tran_bert_attention_mask = []
            tran_text_image_mask = []
            for length in data_translation_length:
                text_mask_cell = [1] * length
                text_mask_cell.extend([0] * (max_translation_length - length))
                tran_bert_attention_mask.append(text_mask_cell[:])

                text_mask_cell.extend([1] * self.image_mask_num)
                tran_text_image_mask.append(text_mask_cell[:])

            temp_labels = [label - 0, label - 1, label - 2]
            target_labels = []
            for i in range(3):
                temp_target_labels = []
                for j in range(temp_labels[0].size(0)):
                    if temp_labels[i][j] == 0:
                        temp_target_labels.append(j)
                target_labels.append(torch.LongTensor(temp_target_labels[:]))

            text_target_labels = []
            if len(text_labels) > 0:
                temp_labels = [text_labels - 0, text_labels - 1, text_labels - 2]
                for i in range(3):
                    temp_target_labels = []
                    for j in range(temp_labels[0].size(0)):
                        if temp_labels[i][j] == 0:
                            temp_target_labels.append(j)
                    text_target_labels.append(torch.LongTensor(temp_target_labels[:]))
                text_target_labels.append(text_labels)

            image_target_labels = []
            if len(image_labels) > 0:
                temp_labels = [image_labels - 0, image_labels - 1, image_labels - 2]
                for i in range(3):
                    temp_target_labels = []
                    for j in range(temp_labels[0].size(0)):
                        if temp_labels[i][j] == 0:
                            temp_target_labels.append(j)
                    image_target_labels.append(torch.LongTensor(temp_target_labels[:]))
                image_target_labels.append(image_labels)

            return text_to_id, torch.LongTensor(bert_attention_mask), image_origin, torch.LongTensor(text_image_mask), label, \
                text_translation_to_id, torch.LongTensor(tran_bert_attention_mask), image_augment, torch.LongTensor(tran_text_image_mask), target_labels, origin_indexes,\
                    text_target_labels, image_target_labels
        else:
            batch_seq_len = []
            batch_graph = []
            batch_bert_indices = []
            batch_attention_mask = []
            batch_ti_attention_mask = []
            batch_boxes = []
            batch_label = []

            batch_bert_indices_aug = []
            batch_graph_aug = []
            batch_attention_mask_aug = []
            batch_seq_len_aug = []

            #self.text_to_id[index], image_origin, self.label_list[index], box_tensors, graph, image_graph,\
            # id_, augt, augg,
            bert_indices_max_len = max([len(t[0]) for t in batch_data])
            bert_indices_max_len_aug = max([len(t[7]) for t in batch_data])
            #text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
            image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
            #label = torch.LongTensor([b[2] for b in batch_data])
            #text_translation_to_id = [torch.LongTensor(b[3]) for b in batch_data]
            #graph = [b[4] for b in batch_data]
            #image_graph = [b[5] for b in batch_data]
            origin_indexes = [b[6] for b in batch_data]

            for item in batch_data: 
                bert_indices, label, boxes, graph, image_graph, aug_bert_indices, aug_graph, aug_image_graph = \
                    item[0], item[2], item[3],item[4], item[5], item[7], item[8], item[9]

                if len(boxes) < 10:
                    for _ in range(10-len(boxes)):
                        boxes.append(torch.zeros_like(boxes[0]))
                else:
                    boxes = boxes[:10]
                batch_boxes.append(torch.stack(boxes, dim=0))
                ###aug
                # aug_image_graph = item['aug_image_graph']
                # aug_graph, aug_box_vit, aug_bert_indices = \
                #     item['aug_graph'],item['aug_box_vit'],item['aug_bert_indices']

                if graph.shape[0] < bert_indices_max_len:
                    graph = np.pad(graph, ((0,bert_indices_max_len-graph.shape[0]),\
                        (0,bert_indices_max_len-graph.shape[0])), 'constant')

                ##aug
                #aug_image_graph = item['aug_image_graph']
                if aug_graph.shape[0] < bert_indices_max_len_aug:
                    aug_graph = np.pad(aug_graph, ((0,bert_indices_max_len_aug-aug_graph.shape[0]),\
                        (0,bert_indices_max_len_aug-aug_graph.shape[0])), 'constant')

                pad_len = image_graph.shape[1]

                graph = np.pad(graph,((0,pad_len),(0,pad_len)),'constant')

                for i in range(image_graph.shape[0]):
                    for j in range(image_graph.shape[1]):
                        if i != 0 and i != image_graph.shape[0]-1:
                            graph[i][j+bert_indices_max_len] = image_graph[i-1][j] + 1
                            graph[j+bert_indices_max_len][i] = image_graph[i-1][j] + 1

                        else:
                            #pass
                            graph[i][j+bert_indices_max_len] =  1
                            graph[j+bert_indices_max_len][i] =  1

                ###aug
                aug_graph = np.pad(aug_graph,((0,pad_len),(0,pad_len)),'constant')
                for i in range(aug_image_graph.shape[0]):
                    for j in range(aug_image_graph.shape[1]):
                        if i != 0 and i != aug_image_graph.shape[0]-1:
                            ##aug
                            aug_graph[i][j+bert_indices_max_len_aug] = aug_image_graph[i-1][j] + 1
                            aug_graph[j+bert_indices_max_len_aug][i] = aug_image_graph[i-1][j] + 1
                        else:
                            ##aug
                            aug_graph[i][j+bert_indices_max_len_aug] =  1
                            aug_graph[j+bert_indices_max_len_aug][i] =  1

                for i in range(image_graph.shape[1]):
                    graph[i+bert_indices_max_len][i+bert_indices_max_len] = 1
                    ##aug
                    aug_graph[i+bert_indices_max_len_aug][i+bert_indices_max_len_aug] = 1


                batch_seq_len.append(len(bert_indices))
                batch_graph.append(graph)

                batch_bert_indices.append(np.pad(bert_indices,(0, bert_indices_max_len - len(bert_indices)),'constant'))
                #batch_box_indices.append(new_box_indices)
                batch_label.append(label)


                ### aug
                batch_seq_len_aug.append(len(aug_bert_indices))
                batch_graph_aug.append(aug_graph)
                batch_bert_indices_aug.append(np.pad(aug_bert_indices,(0, bert_indices_max_len_aug - len(aug_bert_indices)),'constant'))
                # t = [x for x in aug_box_vit]
                # while len(t) < pad_len:
                #     t.append(numpy.zeros(768))
                # aug_batch_box_vit.append(numpy.array(t))


            batch_label = torch.tensor(batch_label)
            temp_labels = [batch_label - 0, batch_label - 1]
            target_labels = []
            for i in range(2):
                temp_target_labels = []
                for j in range(temp_labels[0].size(0)):
                    if temp_labels[i][j] == 0:
                        temp_target_labels.append(j)
                target_labels.append(torch.LongTensor(temp_target_labels[:]))

            for length in batch_seq_len:
                text_mask_cell = [1] * length
                text_mask_cell.extend([0] * (bert_indices_max_len - length))
                # ti_mask = text_mask_cell[:]
                # ti_mask.extend([1] * pad_len)
                ti_mask = [1] * (bert_indices_max_len+pad_len)
                batch_attention_mask.append(text_mask_cell[:])
                batch_ti_attention_mask.append(ti_mask[:])

            ###aug
            for length in batch_seq_len_aug:
                text_mask_cell = [1] * length
                text_mask_cell.extend([0] * (bert_indices_max_len_aug - length))
                # ti_mask = text_mask_cell[:]
                # ti_mask.extend([1] * pad_len)
                ti_mask = [1] * (bert_indices_max_len+pad_len)
                batch_attention_mask_aug.append(text_mask_cell[:])
                #batch_ti_attention_mask.append(ti_mask[:])

            return torch.tensor(np.array(batch_bert_indices)), torch.FloatTensor(batch_attention_mask), image_origin,\
                batch_label, batch_boxes, torch.tensor(np.array(batch_graph)), torch.FloatTensor(batch_ti_attention_mask),\
                    target_labels, origin_indexes, torch.tensor(np.array(batch_bert_indices_aug)), torch.tensor(np.array(batch_graph_aug)), torch.FloatTensor(batch_attention_mask_aug)


def get_resize(image_size):
    for i in range(20):
        if 2**i >= image_size:
            return 2**i
    return image_size


def data_process(opt, data_path, text_tokenizer, photo_path, data_type, data_translation_path=None, image_coordinate=None, distributed=0, text_image_labels_path=None, gcn_root=None):
    mean_resnet = [0.485, 0.456, 0.406]
    std_resnet = [0.229, 0.224, 0.225]
    mean_vit = [0.5, 0.5, 0.5]
    std_vit = [0.5, 0.5, 0.5]
    mean_clip = [0.48145466,0.4578275,0.40821073]
    std_clip = [0.26862954,0.26130258,0.27577711] 
    mean_albef = [0.48145466, 0.4578275, 0.40821073]
    std_albef = [0.26862954, 0.26130258, 0.27577711]
 
    if 'resnet' in opt.image_model:
        mean, std = mean_resnet, std_resnet
        interpolation = 2
    elif 'vit' in opt.image_model:
        mean, std = mean_vit, std_vit
        interpolation = 2
    elif 'clip' in opt.image_model:
        mean, std = mean_clip, std_clip
        interpolation = 3
    elif 'ALBEF' in opt.image_model:
        mean, std = mean_albef, std_albef
        interpolation = 3

    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size),interpolation=interpolation),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size),interpolation=interpolation),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    transform_augment = copy.deepcopy(transform_base)
    transform_augment.transforms.insert(0, RandAugment(2, 14))
    transform_train =  transform_test_dev #transform_augment


    dataset = SentenceDataset(opt, data_path, text_tokenizer, photo_path, transform_train if (data_type == 1 and not opt.gcn) else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate, text_image_labels_path=text_image_labels_path, gcn_root=gcn_root)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if ( not distributed and data_type == 1 ) else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), 
                             pin_memory=True if opt.cuda else False,
                             sampler=DistributedSampler(dataset) if (distributed and data_type==1) else None)
    return data_loader, dataset.__len__()

