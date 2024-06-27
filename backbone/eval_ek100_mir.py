import torch

from model.setup_model import *
from data.ek100_dataset import EK100Dataset
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
import torchvision
from util.config import *
import argparse
import pandas as pd
from util.meter import get_mAP,get_nDCG

def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION pretrain', add_help=False)
    ## Data ##
    parser.add_argument('--config', default='configs/demo_ek100_mir.yaml', type=str)
    return parser
parser = argparse.ArgumentParser('EgoVideo model evaluation', parents=[get_args_parser()])
args = parser.parse_args()

cfg = get_config(args)
# load model ckpt
model,tokenizer = build_model(ckpt_path = cfg.ckpt_path,num_frames = cfg.num_frames)

model = model.eval()
# build dataloader
class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)

crop_size = 224
mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]  
base_val_transform_ls = [
    Permute([3, 0, 1, 2]),
    torchvision.transforms.Resize(crop_size),
    torchvision.transforms.CenterCrop(crop_size),
    transforms_video.NormalizeVideo(mean=mean, std=std),
]
val_transform = torchvision.transforms.Compose(base_val_transform_ls)

val_dataset = EK100Dataset(cfg.ek100_mir, transform=val_transform, is_training=False, tokenizer=tokenizer, crop_size=crop_size)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
)
# switch to eval mode
model = model.eval().to('cuda').to(torch.float16)

all_video_embed = []
all_text_embed = []
total_num = 0

with torch.no_grad():
    for i, inputs in enumerate(val_loader):
        # measure data loading time
        inputs = [tensor.to('cuda') for tensor in inputs]
        _ = inputs.pop()  # loader will a "relevancy" variable which is not needed except ek100_mir
        inputs[0] = inputs[0].to(torch.float16)
        # compute output    
        image_features, text_features = model(*inputs)

        all_video_embed.append(image_features)
        all_text_embed.append(text_features)

        if i % 100 == 0:
            print(f'process {i}/{len(val_loader)}')

all_video_embed = torch.stack(all_video_embed).squeeze().cpu().numpy()  
all_text_embed = torch.stack(all_text_embed).squeeze().cpu().numpy()

similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
similarity_matrix = (similarity_matrix + 1) / 2

video_id = pd.read_csv(cfg.ek100_mir.metadata).values[:, 0]
text_id = pd.read_csv(cfg.ek100_mir.metadata.replace('test', 'test_sentence')).values[:, 0]
indexes = [video_id.tolist().index(elem) for elem in text_id]
similarity_matrix = similarity_matrix[:, indexes]
print(similarity_matrix.shape,text_id.shape,video_id.shape)
rel_matrix = pd.read_pickle(cfg.ek100_mir.relevancy_path)
vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(similarity_matrix, rel_matrix)
print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))


import json
wewant = {'sim_mat':similarity_matrix.tolist(),'map':avg_map,'nDCG':avg_nDCG}
with open(f'results/zs_mir_result.json','w') as f:
        json.dump(wewant,f)
        f.close()