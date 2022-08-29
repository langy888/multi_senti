"""
Name: train_process
Date: 2022/4/11 上午10:26
Version: 1.0

"""

import torch
# from transformers import AdamW
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
import dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
# import tensorflow as tf
import torch.distributed as dist
from time import time


def is_main_process() -> bool:
    return dist.get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion, log_summary_writer:SummaryWriter=None, tokenizer=None, image_id_list=None):
    optimizer = None
    training_patience = 7
    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    pre_img_model_param = [name for name, param in cl_model.named_parameters() if 'image_model' in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
            "lr": 0,
        },
        {
            "params": [p for n, p in cl_model.named_parameters() if (n not in pre_train_model_param and n not in pre_img_model_param)],
            "lr": opt.fuse_lr,
        },
        # {
        #     "params": [p for n, p in cl_model.named_parameters() if n in pre_img_model_param],
        #     "lr": 2e-5,
        # },
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    orgin_param = ModelParam()
    augment_param = ModelParam()

    last_F1 = 0
    last_Accuracy = 0
    Best_Acc = 0
    Best_F1 = 0
    for epoch in trange(opt.epoch, desc='Epoch:'):
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0

        cl_model.train()
        cl_model.zero_grad()

        if epoch >= opt.train_fuse_model_epoch:
            optimizer.param_groups[0]['lr'] = opt.lr
            optimizer.param_groups[1]['lr'] = opt.lr #1e-3   #opt.lr
            #optimizer.param_groups[2]['lr'] = opt.lr

        if  opt.cuda and is_main_process():
            train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        else:
            train_loader_tqdm = train_loader

        epoch_step_num = epoch * len(train_loader)
        step_num = 0
        for index, data in enumerate(train_loader_tqdm):
            if opt.gcn:

                texts_origin, bert_attention_mask, image_origin, labels, batch_boxes,\
                    graphs, text_image_mask, target_labels, _, aug_text, aug_graph, aug_mask = data

                if opt.cuda is True:
                    texts_origin = texts_origin.cuda()

                    ##aug
                    aug_text = aug_text.cuda()
                    aug_mask = aug_mask.cuda()
                    aug_graph = aug_graph.cuda()

                    bert_attention_mask = bert_attention_mask.cuda()
                    image_origin = image_origin.cuda()
                    text_image_mask = text_image_mask.cuda()
                    labels = labels.cuda()
                    graphs = graphs.cuda()
                    for i in range(len(target_labels)):
                        target_labels[i] = target_labels[i].cuda()
                    for i in range(len(batch_boxes)):
                        batch_boxes[i] = batch_boxes[i].cuda()


                orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask, boxes=batch_boxes, graph=graphs)
                augment_param.set_data_param(texts=aug_text, bert_attention_mask=aug_mask, images=image_origin, text_image_mask=text_image_mask,  boxes=batch_boxes, graph=aug_graph)

                origin_res, other_loss = cl_model(orgin_param, augment_param, labels, target_labels)

                #stl, sil, atl, ail = other_loss
                #sum_other_loss = stl*1 + sil*1 + atl + ail
                #+ sum_other_loss

                classify_loss = critertion(origin_res, labels)
                cl_self_loss, l_pos_neg, cl_lables = other_loss
                cl_loss = critertion(l_pos_neg, cl_lables) # 原始 和 aug 相似度 分类loss
                #loss = classify_loss + ( cl_self_loss / opt.acc_batch_size )
                loss = (classify_loss * opt.cls_loss_alpha + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha ) / opt.acc_batch_size

            else:
                texts_origin, bert_attention_mask, image_origin, text_image_mask, labels,\
                    texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, origin_indexes,\
                    text_target_labels , image_target_labels = data

                if opt.cuda is True:
                    texts_origin = texts_origin.cuda()
                    bert_attention_mask = bert_attention_mask.cuda()
                    image_origin = image_origin.cuda()
                    text_image_mask = text_image_mask.cuda()
                    labels = labels.cuda()
                    texts_augment = texts_augment.cuda()
                    bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                    image_augment = image_augment.cuda()
                    text_image_mask_augment = text_image_mask_augment.cuda()
                    for i in range(len(target_labels)):
                        target_labels[i] = target_labels[i].cuda()
                    if text_target_labels:
                        for i in range(len(text_target_labels)):
                            text_target_labels[i] = text_target_labels[i].cuda()
                        for i in range(len(image_target_labels)):
                            image_target_labels[i] = image_target_labels[i].cuda()

                orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask)
                augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)

                origin_res, l_pos_neg, cl_lables, cl_self_loss, other_loss = cl_model(orgin_param, augment_param, labels, target_labels, text_sep_labels=text_target_labels, image_sep_labels=image_target_labels)

                #stl, sil, atl, ail = other_loss
                #sum_other_loss = stl*1 + sil*1 + atl + ail
                #+ sum_other_loss

                classify_loss = critertion(origin_res, labels)
                if cl_lables is not None:
                    cl_loss = critertion(l_pos_neg, cl_lables) # 原始 和 aug 相似度 分类loss

                loss = classify_loss + cl_self_loss / opt.acc_batch_size  #(classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha ) / opt.acc_batch_size

            
            loss.backward()
            if opt.cuda and dist.get_rank() == 0 :
                train_loader_tqdm.set_description("Train Iteration, loss: %.6f, lr: %e" %
                                                (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_num += 1

            _, predicted = torch.max(origin_res, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1, train_precision, train_R, run_loss)
        if is_main_process():
            WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
            print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_w', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/r_w', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/p_w', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_ma', train_F1, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "train_F1_weighted": train_F1_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_R_weighted": train_R_weighted,
            "run_loss": run_loss
        }
        # debug：正常运行不要把下面的代码注释掉
        if epoch %1 == 0 and is_main_process():
            print(f"GPU: {dist.get_rank()} eval")
            last_F1, last_Accuracy = dev_process.dev_process(opt, critertion, cl_model.module, dev_loader, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)
        if last_F1 > Best_F1:
            Best_F1 = last_F1
            training_patience = 7

        if last_Accuracy > Best_Acc:
            Best_Acc = last_Accuracy
            training_patience = 7

        if training_patience == 0:
            print("Stop Training due to no improvement")
            exit()

        if training_patience != 7:
            training_patience -= 1 

        print(f"GPU: {dist.get_rank()} barriered")
        dist.barrier()
        print(f"GPU: {dist.get_rank()} out barriered")
