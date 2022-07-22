export http_proxy=172.16.1.135:3128 && export https_proxy=172.16.1.135:3128
gpu_num=$2
config=$1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#HF_DATASETS_OFFLINE=1 
#TRANSFORMERS_OFFLINE=1
#python main.py -config_path $config -gpu_num $gpu_num
#export OMP_NUM_THREADS=1 
#python -m torch.distributed.launch --nproc_per_node=2 main_dist.py -config_path $config -gpu_num $gpu_num
python main_dist.py -config_path $config -gpu_num $gpu_num