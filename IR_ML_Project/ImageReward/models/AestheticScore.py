'''
@File       :   AestheticScore.py
@Time       :   2023/02/12 14:54:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   AestheticScore.
* Based on improved-aesthetic-predictor code base
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip



class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        self.mlp = MLP(768)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) 

        
        self.clip_model.logit_scale.requires_grad_(False)
        
    def score(self, prompt, image_path):
        
        if (type(image_path).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image_path)
            return rewards
            
        
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image)).float()
        
        
        rewards = self.mlp(image_features)
        
        return rewards.detach().cpu().numpy().item()

    def inference_rank(self, prompt, generations_list):
        
        img_set = []
        for generations in generations_list:
            
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)
            
        img_features = torch.cat(img_set, 0).float() 
        rewards = self.mlp(img_features)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()