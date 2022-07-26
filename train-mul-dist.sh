export http_proxy=172.16.1.135:3128 && export https_proxy=172.16.1.135:3128
export TORCH_HOME=/output/pretrained_model/input
lr=2e-5
fuse_lr=2e-5
image_output_type=all
optim=adamw
data_type=MVSA-multiple
train_fuse_model_epoch=20
epoch=30
warmup_step_epoch=2
fuse_type=att
batch=64
acc_grad=1
tran_num_layers=4
image_num_layers=2


fixed_image_model=1
no_extra_img_trans=0


text_model=simces_roberta_sup
image_model=resnet-50
#bert-base
#roberta_base 
#resnet-50
#vit_b_16
#clip_vit_b
#simces_roberta_sup
#simces_roberta_unsup

run_type=1
test_model_path=/output/checkpoint/2022-07-20-17-17-15-MVSA-multiple-4-att-64-2-roberta_base-/07-20-20-11-26-Acc-0.70529.pth

python -m torch.distributed.launch --nproc_per_node=$1 --nnodes=1 main_dist.py -cuda -gpu_num $1 -epoch ${epoch} -num_workers 0  \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model}-${image_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model ${image_model} -run_type ${run_type}  -test_model_path ${test_model_path}\
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type} -fixed_image_model ${fixed_image_model} \
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch} \
        -no_extra_img_trans ${no_extra_img_trans} 
