import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DinoWithLinearHead(nn.Module):
    def __init__(self, pth_path):
        super().__init__()
        self.head = nn.Linear(384, 10)
        self.pth_path_global = os.path.abspath(os.path.join(os.path.dirname(__file__), pth_path))
        self._load_head(self.pth_path_global)
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
        self.dino = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')

    def _load_head(self, pth_path):
        if os.path.exists(pth_path):
            state_dict = torch.load(pth_path)
            if 'model' in state_dict:
                self.head.load_state_dict(state_dict['model'])
            else:
                self.head.load_state_dict(state_dict)
        else:
            print("No file found")
            import glob
            checkpoints = glob.glob('../../models/linear_head_classification/checkpoint_epoch_*.pth')
            latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(latest)
            self.head.load_state_dict(checkpoint['model'])

    def forward(self, img):
        dino_inputs = self.processor(img, return_tensors='pt')
        outputs = self.dino(**dino_inputs)
        features = outputs.last_hidden_state[:, 0, :]
        logits = self.head(features)
        return logits

