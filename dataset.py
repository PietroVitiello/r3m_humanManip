import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import glob

from sentence_transformers import SentenceTransformer


class ImageActionsDataset(torch.utils.data.Dataset):
    def __init__(self, path='data/train'):
        self.dataset_path = path
        self._embed = SentenceTransformer("all-MiniLM-L6-v2")
  
    def __len__(self):
        data_path = self.dataset_path + "/episode_*.npy"
        episode_paths = glob.glob(data_path)
        return len(episode_paths)
  
    def __getitem__(self, idx):
        # load raw data --> this should change for your dataset
        episode_path = self.dataset_path + "/episode_{}.npy".format(idx)
        data = np.load(episode_path, allow_pickle=True).item()   # this is a list of dicts in our case

        rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        rgb[80:560] = data['image']
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = transforms.Resize(256)(torch.tensor(rgb))

        actions = np.zeros((40, 4, 3), dtype=np.float32)
        actions[:len(data['actions'])] = data['actions']
        actions = actions.reshape(-1, 12) # to undo actions.reshape((40, 12))

        lang_embedding = self._embed.encode([data['language_instruction']])[0]
        # print(f"lang_embedding: {lang_embedding.shape}")

        return {
            "obs": rgb, # (3, 224, 224)
            "lang": torch.tensor(lang_embedding), # (384,)
            "action" : torch.tensor(actions), # (40, 12)
        }
    
        #return {"obs": self.inputs[x].reshape((1,-1)), "action" : self.outputs[x]}


class ImageActionsDataset_Unaltered(torch.utils.data.Dataset):
    def __init__(self, path='data/train'):
        self.dataset_path = path
        self._embed = SentenceTransformer("all-MiniLM-L6-v2")
  
    def __len__(self):
        data_path = self.dataset_path + "/episode_*.npy"
        episode_paths = glob.glob(data_path)
        return len(episode_paths)
  
    def __getitem__(self, idx):
        # load raw data --> this should change for your dataset
        episode_path = self.dataset_path + "/episode_{}.npy".format(idx)
        data = np.load(episode_path, allow_pickle=True).item()   # this is a list of dicts in our case
        actions = np.zeros((40, 4, 3), dtype=np.float32)
        actions[:len(data['actions'])] = data['actions']
        data['actions'] = actions.reshape(-1, 12) # to undo actions.reshape((40, 12))
        return data
