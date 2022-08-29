export http_proxy=172.16.1.135:3128 && export https_proxy=172.16.1.135:3128
export TORCH_HOME=/output/pretrained_model/input
lr=2e-5
fuse_lr=2e-5
image_output_type=all
optim=adamw
data_type=MVSA-single
train_fuse_model_epoch=20
epoch=30
warmup_step_epoch=2
text_model=bert-base
fuse_type=att
batch=32
acc_grad=1
tran_num_layers=3
image_num_layers=2

run_type=2
test_model_path=/output/checkpoint/2022-08-04-13-15-29-MVSA-single-3-att-32-2-bert-base-/08-04-13-46-18-Acc-0.74444.pth

python  main_dist.py -cuda -gpu_num $1 -epoch ${epoch} -num_workers 0 \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model resnet-50 \
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type} -fixed_image_model 1 \
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch}
