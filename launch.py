import torch.multiprocessing as mp
import torch.distributed as dist
from main import main
import argparse
from config.args import add_parser
from util.load_config import load_config

parse = argparse.ArgumentParser()
parse.add_argument('-config_path', type=str,
                    default='', help='path of config file')
parse.add_argument('-gpu_num', type=int, default=1, help='gpu nums')
parse.add_argument('--nodes', type=int, default=1)
parse.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parse = add_parser(parse)
args = parse.parse_args()
if args.config_path != "":
    config = load_config(args.config_path)
parse.set_defaults(**config)
opt = parse.parse_args()
opt.world_size = opt.gpu_num * opt.nodes
mp.spawn(main, nprocs=opt.gpus, args=(opt,))