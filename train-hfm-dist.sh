export http_proxy=172.16.1.135:3128 && export https_proxy=172.16.1.135:3128
export TORCH_HOME=/output/pretrained_model/input

lr=2e-5
fuse_lr=2e-5
image_output_type=all
optim=adamw
data_type=HFM
train_fuse_model_epoch=0
epoch=50
warmup_step_epoch=2
text_model=bert-base
fuse_type=att
batch=32
acc_grad=1
tran_num_layers=5
image_num_layers=1

fixed_image_model=0
no_extra_img_trans=0
gcn=0

text_model=bert-base
image_model=resnet-50
#bert-base
#roberta_base 
#resnet-50
#vit_b_16
#clip_vit_b
#simces_roberta_sup
#simces_roberta_unsup

cls_loss_fac=1
cl_loss_aug_fac=1
cl_loss_label_fac=1

run_type=1
test_model_path=/output/checkpoint/

python -m torch.distributed.launch --nproc_per_node=$1 --nnodes=1 main_dist.py -cuda -gpu_num $1 -epoch ${epoch} -num_workers 0  \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model}-${image_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model ${image_model} -run_type ${run_type}  -test_model_path ${test_model_path}\
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type} -fixed_image_model ${fixed_image_model} \
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch} \
        -no_extra_img_trans ${no_extra_img_trans} -gcn ${gcn}  -cl_loss_alpha ${cl_loss_aug_fac} -cl_self_loss_alpha ${cl_loss_label_fac} -cls_loss_alpha ${cls_loss_fac}