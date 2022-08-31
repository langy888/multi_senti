"""
Name: data_process
Date: 2022/4/11 上午10:25
Version: 1.0
"""

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

# debug
# flagflag=0


class SentenceDataset(Dataset):
    def __init__(self, opt, data_path, text_tokenizer, photo_path, image_transforms, data_type, data_translation_path=None, image_coordinate=None):
        self.data_type = data_type
        self.dataset_type = opt.data_type
        self.photo_path = photo_path
        self.image_transforms = image_transforms

        file_content = json.load(open(data_path, 'r', encoding='utf-8'))

        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        self.emoji_list = []
        self.hashtag_list = []
        for data in file_content:
            self.data_id_list.append(data['id'])
            self.text_list.append(data['text'])
            self.label_list.append(data['emotion_label'])
            self.emoji_list.append(data['emoji'])
            self.hashtag_list.append(data['hashtag'])


        self.image_id_list = self.data_id_list

        file_read = open(data_translation_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()
        self.aug_data_id2text = {data['id']: data['text_translation'] for data in file_content}

        # for index, id_ in enumerate(self.data_id_list):
        #     text = self.text_list[index]
        #     aug_text = self.

        ## text
        self.text_token_list = [text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in tqdm(self.text_list, desc='convert text to token')]
        self.aug_text_tokens = {index: text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for index, text in self.aug_data_id2text.items()}
        self.text_token_list = [text if len(text) < opt.max_token_length else text[0: opt.max_token_length] for text in
                                self.text_token_list]
        self.text_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in
                           tqdm(self.text_token_list, desc='convert text to id')]
        self.aug_text_tokens = {index: text_token if len(text_token) < opt.max_token_length else text_token[0:opt.max_token_length] for index, text_token in
                                                  self.aug_text_tokens.items()}
        self.aug_text_ids = {index: text_tokenizer.convert_tokens_to_ids(text_token) for index, text_token in self.aug_text_tokens.items()}

        ##emoji
        self.emoji_token_list = [text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in tqdm(self.emoji_list, desc='convert emoji to token')]
        self.emoji_token_list = [text if len(text) < opt.max_else_length else text[0: opt.max_else_length] for text in
                                self.emoji_token_list]
        self.emoji_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in
                           tqdm(self.emoji_token_list, desc='convert emoji to id')]
      
        ##hashtag
        self.hashtag_token_list = [text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in tqdm(self.hashtag_list, desc='convert hashtag to token')]
        self.hashtag_token_list = [text if len(text) < opt.max_else_length else text[0: opt.max_else_length] for text in
                                self.hashtag_token_list]
        self.hashtag_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in
                           tqdm(self.hashtag_token_list, desc='convert hashtag to id')]

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        image_path = self.photo_path + '/' + str(self.data_id_list[index]) + '.jpg'
        image_read = Image.open(image_path)
        image_read.load()

        image_origin = self.image_transforms(image_read)
        image_augment = image_origin
        if self.data_type == 1:
            image_augment = copy.deepcopy(image_read)
            image_augment = self.image_transforms(image_augment)
        return self.text_to_id[index], image_origin, self.label_list[index], self.aug_text_ids[self.data_id_list[index]], image_augment,\
            self.emoji_to_id[index], self.hashtag_to_id[index]


class Collate():
    def __init__(self, opt):
        self.text_length_dynamic = opt.text_length_dynamic
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.max_token_length

        self.image_mask_num = 0
        if opt.image_output_type == 'cls':
            self.image_mask_num = 1
        elif opt.image_output_type == 'all':
            self.image_mask_num = 50

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
        image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
        label = torch.LongTensor([b[2] for b in batch_data])
        aug_text_ids = [torch.LongTensor(b[3]) for b in batch_data]
        image_augment = torch.FloatTensor([np.array(b[4]) for b in batch_data])
        emoji_ids = [torch.LongTensor(b[5]) for b in batch_data]
        hashtag_ids = [torch.LongTensor(b[6]) for b in batch_data]

        data_len = [text.size(0) for text in text_to_id]
        aug_data_len = [text.size(0) for text in aug_text_ids]

        max_len = max(data_len)
        if max_len < self.min_length:
            text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
            max_len = self.min_length

        aug_max_len = max(aug_data_len)
        if aug_max_len < self.min_length:
            aug_text_ids[0] = torch.cat((aug_text_ids[0], torch.LongTensor([0] * (self.min_length - aug_text_ids[0].size(0)))))
            aug_max_len = self.min_length

        emoji_len = [text.size(0) for text in emoji_ids]
        hashtag_len = [text.size(0) for text in hashtag_ids]

        max_emoji_len = max(emoji_len)
        max_hashtag_len = max(hashtag_len)
        emoji_ids = run_utils.pad_sequence(emoji_ids, batch_first=True, padding_value=0)
        hashtag_ids = run_utils.pad_sequence(hashtag_ids, batch_first=True, padding_value=0)


        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
        aug_text_ids = run_utils.pad_sequence(aug_text_ids, batch_first=True, padding_value=0)

        bert_attention_mask = []
        text_image_mask = []
        for length, el, htl in zip(data_len, emoji_len, hashtag_len):
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_len - length))
            #text_mask_cell.extend([1] * el)
            #text_mask_cell.extend([0] * (max_emoji_len - el))
            #text_mask_cell.extend([1] * htl)
            #text_mask_cell.extend([0] * (max_hashtag_len - htl))
            bert_attention_mask.append(text_mask_cell[:])

            text_mask_cell.extend([1] * self.image_mask_num)
            text_image_mask.append(text_mask_cell[:])

        tran_bert_attention_mask = []
        tran_text_image_mask = []
        for length, el, htl in zip(aug_data_len, emoji_len, hashtag_len):
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (aug_max_len - length))
            #text_mask_cell.extend([1] * el)
            #text_mask_cell.extend([0] * (max_emoji_len - el))
            #text_mask_cell.extend([1] * htl)
            #text_mask_cell.extend([0] * (max_hashtag_len - htl))
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

        return text_to_id, torch.LongTensor(bert_attention_mask), image_origin, torch.LongTensor(text_image_mask), label, \
               aug_text_ids, torch.LongTensor(tran_bert_attention_mask), image_augment, torch.LongTensor(tran_text_image_mask), target_labels, \
                emoji_ids, hashtag_ids


def get_resize(image_size):
    for i in range(20):
        if 2**i >= image_size:
            return 2**i
    return image_size


def data_process(opt, data_path, text_tokenizer, photo_path, data_type, data_translation_path=None, image_coordinate=None, distributed=0):

    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # transform_train = copy.deepcopy(transform_base)
    transform_augment = copy.deepcopy(transform_base)
    transform_augment.transforms.insert(0, RandAugment(2, 14))
    transform_train = transform_augment
    # transform_train = [transform_train, transform_augment]

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = SentenceDataset(opt, data_path, text_tokenizer, photo_path, transform_train if data_type == 1 else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if ( not distributed and data_type == 1 ) else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), 
                             pin_memory=True if opt.cuda else False,
                             sampler=DistributedSampler(dataset) if (distributed and data_type==1) else None)
    return data_loader, dataset.__len__()

