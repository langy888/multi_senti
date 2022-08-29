import torch
import os 
os.environ["TORCH_HOME"] = '/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/pretrained_model/input'
import torchvision
from transformers import BertConfig, BertForPreTraining, AutoTokenizer, AutoModel,\
    ViTConfig, ViTModel, ViTFeatureExtractor
import sys
sys.path.append("/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF")
#fe = ViTFeatureExtractor.from_pretrained('/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/pretrained_model/vit_b_16')

# vit = ViTModel.from_pretrained('/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/pretrained_model/vit_b_16')
# images = torch.randint(0,255,(3, 32, 32))
# px = fe(images, 'pt')
# outputs = vit(**px)
# print(outputs.last_hidden_state.shape, outputs.pooler_output.shape)
# #for param in vit.parameters():
# #    print()
# print(vit.config.hidden_size) 

#model = torchvision.models.
#print(torchvision.__version__)

#from ..ALBEF.models.xbert import BertConfig, BertModel
from transformers import AutoTokenizer

model = torch.load('/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/pretrained_model/ALBEF/ALBEF_4M.pth', map_location='cpu')
print()