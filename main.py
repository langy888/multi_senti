"""
Name: main
Date: 2022/4/11 上午10:25
Version: 1.0
"""

import os
import argparse
import data_process
# import train_process_debug as train_process
import train_process
import torch
import torch.nn.modules as nn
import dev_process
import test_process
import model
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
from util.write_file import WriteFile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
import torch.distributed as dist
from config.args import add_parser
from util.load_config import load_config

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-config_path', type=str,
                       default='', help='path of config file')
    parse.add_argument('-gpu_num', type=int, default=1, help='gpu nums')
    parse = add_parser(parse)
    args = parse.parse_args()
    #config = load_config("base_config.yaml")
    if args.config_path != "":
        config = load_config(args.config_path)
        #config = merge_config(config, temp_config)
    parse.set_defaults(**config)
    opt = parse.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)
    opt.epoch = opt.train_fuse_model_epoch + opt.epoch

    dt = datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    # CE（交叉熵），SCE（标签平滑的交叉熵），Focal（focal loss），Ghm（ghm loss）
    critertion = None
    if opt.loss_type == 'CE':
        critertion = nn.CrossEntropyLoss()

    cl_fuse_model = model.CLModel(opt)
    if opt.cuda is True:
        assert torch.cuda.is_available()
        if opt.gpu_num > 1:
            if opt.gpu0_bsz > 0:
                cl_fuse_model = torch.nn.DataParallel(cl_fuse_model).cuda()
            else:
                opt.local_rank = 0
                print('multi-gpu')
                """
                单机多卡的运行方式，nproc_per_node表示使用的GPU的数量
                python -m torch.distributed.launch --nproc_per_node=2 main.py
                """
                print('当前GPU编号：', opt.local_rank)
                torch.cuda.set_device(opt.local_rank) # 在进行其他操作之前必须先设置这个
                torch.distributed.init_process_group(backend='nccl')
                cl_fuse_model = cl_fuse_model.cuda()
                cl_fuse_model = nn.parallel.DistributedDataParallel(cl_fuse_model, find_unused_parameters=True)
        else:
            cl_fuse_model = cl_fuse_model.cuda()
        critertion = critertion.cuda()

    print('Init Data Process:')
    tokenizer = None
    abl_path = ''
    if opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/vocab.txt')

    if opt.data_type == 'HFM':
        data_path_root = abl_path + 'dataset/data/HFM/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'valid.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = data_path_root + '/dataset_image'
        image_coordinate = None
        data_translation_path = data_path_root + '/HFM.json'
    else:
        data_path_root = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_path_name + '/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'dev.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = abl_path + 'dataset/data/' + opt.data_type + '/dataset_image'
        image_coordinate = None
        data_translation_path = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_type + '_translation.json'

    # data_type:标识数据的类型，1是训练数据，2是开发集，3是测试数据
    train_loader, opt.train_data_len = data_process.data_process(opt, train_data_path, tokenizer, photo_path, data_type=1, data_translation_path=data_translation_path,
                                                                 image_coordinate=image_coordinate)
    dev_loader, opt.dev_data_len = data_process.data_process(opt, dev_data_path, tokenizer, photo_path, data_type=2, data_translation_path=data_translation_path,
                                                             image_coordinate=image_coordinate)
    test_loader, opt.test_data_len = data_process.data_process(opt, test_data_path, tokenizer, photo_path, data_type=3, data_translation_path=data_translation_path,
                                                               image_coordinate=image_coordinate)

    if opt.warmup_step_epoch > 0:
        opt.warmup_step = opt.warmup_step_epoch * len(train_loader)
        opt.warmup_num_lr = opt.lr / opt.warmup_step
    opt.scheduler_step_epoch = opt.epoch - opt.warmup_step_epoch

    if opt.scheduler_step_epoch > 0:
        opt.scheduler_step = opt.scheduler_step_epoch * len(train_loader)
        opt.scheduler_num_lr = opt.lr / opt.scheduler_step

    print(opt)
    opt.save_model_path = WriteFile(opt.save_model_path, 'train_correct_log.txt', str(opt) + '\n\n', 'a+', change_file_name=True)
    log_summary_writer = None
    log_summary_writer = SummaryWriter(log_dir=opt.save_model_path)
    log_summary_writer.add_text('Hyperparameter', str(opt), global_step=1)
    log_summary_writer.flush()

    if opt.run_type == 1:
        print('\nTraining Begin')
        train_process.train_process(opt, train_loader, dev_loader, test_loader, cl_fuse_model, critertion, log_summary_writer)
    elif opt.run_type == 2:
        print('\nTest Begin')
        model_path = "checkpoint/best_model/best-model.pth"
        cl_fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        test_process.test_process(opt, critertion, cl_fuse_model, test_loader, epoch=1)

    log_summary_writer.close()
