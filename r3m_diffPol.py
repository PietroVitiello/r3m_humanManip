from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from dataset import ImageActionsDataset, ImageActionsDataset_Unaltered
from torch.utils.data import DataLoader

import os
import wandb
from r3m import load_r3m
from diffPol import DiffPolicy
from sentence_transformers import SentenceTransformer
from torchvision import transforms



os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VIP_DiffPol(nn.Module):
    def __init__(self,
                 action_dim,
                 action_horizon,
                 device='cuda', **kwargs):
        
        super().__init__()
        self.r3m = load_r3m("resnet18") # resnet18, resnet34
        self.r3m.eval()
        self.diff_pol = DiffPolicy(
            896, action_dim, action_horizon, device=device
        )
        self.device = device
        self._lang_encoder = None


    def eval(self):
        self._lang_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.train(False)

    def save_model(self, name: str, epoch: int, ema: EMAModel = None):
        save_dir = f"models/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = f"{save_dir}/diff_pol_{epoch}.pt"
        if ema is not None:
            torch.save(ema.state_dict(), save_dir)
        else:
            torch.save(self.diff_pol.state_dict(), save_dir)

    def load_model(self, model_fname: str):
        load_dir = f"models/{model_fname}"

        saved_state_dict = torch.load(load_dir)
        self_state_dict = self.diff_pol.state_dict()
        shadow_params = saved_state_dict['shadow_params']

        p_counter = 0
        for name, _ in self_state_dict.items():
            saved_param = shadow_params[p_counter]
            if name in self_state_dict:
                self_state_dict[name].copy_(saved_param)
            else:
                print(f"Parameter {name} not found in PyTorch model.")
            p_counter += 1

    def forward(self, data):
        rgb = data["obs"].to(self.device)
        lang_enc = data["lang"].to(self.device)

        with torch.no_grad():
            vis_enc = self.r3m(rgb)
            # print(f"visual representation: {vis_enc.shape}")
            # print(f"language representation: {lang_enc.shape}")

            obs_enc = torch.concat((vis_enc, lang_enc), dim=1)
            # print(f"observation representation: {obs_enc.shape}")

        actions = data["action"].to(self.device)
        noise, noise_pred = self.diff_pol(obs_enc, actions)

        return noise, noise_pred
    
    def inference(self,
                  rgb: torch.Tensor,
                  language_description: str,
                  noisy_actions: torch.Tensor = None,
                  diffusion_iters: int = None):
        assert rgb.shape == (480, 640, 3), "Input image shape should be (480, 640, 3)"
        assert isinstance(language_description, str), "Language description should be a string"

        rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        rgb[80:560] = data['image']
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = transforms.Resize(256)(torch.tensor(rgb))
        rgb = rgb.to(self.device).unsqueeze(0)

        lang_enc = torch.tensor(self._lang_encoder.encode(
            [language_description]
        )[0]).to(self.device).unsqueeze(0)

        with torch.no_grad():
            vis_enc = self.r3m(rgb)
            obs_enc = torch.concat((vis_enc, lang_enc), dim=1)
            pred_actions = self.diff_pol.inference(
                obs_enc, noisy_actions, diffusion_iters
            )

        return pred_actions
        







    
def train(model: VIP_DiffPol,
          dataloader: DataLoader,
          num_epochs: int,
          run_name: str = "default",
          use_wandb: bool = False,):
    
    if use_wandb:
        run = wandb.init(
            project="RplusX",
            name=run_name,
            tags=["r3mDiff",]
        )

    diff_policy = model.diff_pol

    ema = EMAModel(
            model=diff_policy,
            power=0.75,
            parameters=diff_policy.parameters()
        )

    optimizer = torch.optim.AdamW(
        params=diff_policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    losses = []
    for epoch_idx in tqdm(range(num_epochs), desc='Epochs', leave=True, position=0):
        if epoch_idx%200 == 0:
            print(epoch_idx)
        epoch_loss = []

        for b_idx, b in tqdm(enumerate(dataloader), desc='Batch', leave=False, position=1):
            # data normalized in dataset
            # device transfer

            noise, noise_pred = model(b)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(diff_policy.parameters())

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            if epoch_idx % 100 == 0 and use_wandb: #TODO: change to b_idx
                wandb.log({
                    '[STEP] Loss': loss_cpu,
                    'Train batch': epoch_idx * len(dataloader) + b_idx,
                })
        
        if epoch_idx % 999 == 0 and epoch_idx != 0: #TODO: remove!!!!!!!!!!!
            model.save_model(run_name, epoch_idx, ema)
        if epoch_idx % 300 == 0 and use_wandb: #TODO: remove and swap with line below
        # if use_wandb: #TODO: change to batch_idx
            wandb.log({
                '[EPOCH] Loss': loss_cpu,
                'Train batch': epoch_idx * len(dataloader) + b_idx,
            })

    # Weights of the EMA model
    # is used for inference
    ema_parameters = ema.state_dict()
    model_params_state_dict = diff_policy.state_dict()

    for name, param in ema_parameters.items():
        if name in model_params_state_dict:
            model_params_state_dict[name].copy_(param)
        else:
            print(f"Parameter {name} not found in PyTorch model.")

    diff_policy.load_state_dict(model_params_state_dict)

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    

if __name__ == "__main__":
    model = VIP_DiffPol(action_dim=12, action_horizon=40)


    data = ImageActionsDataset()
    train_dataloader = DataLoader(data, batch_size=8, shuffle=True)

    print(f"Dataset length: {len(data)}\nExample data:")
    print(data[0])
    print(model(next(iter(train_dataloader))))
    train(model, train_dataloader, num_epochs=2000, run_name="try3", use_wandb=True)



    # model_fname = "try3/diff_pol_2997.pt"
    # model.load_model(model_fname)
    # model.eval()

    # data = ImageActionsDataset_Unaltered()
    # data = data[0]
    # rgb = data["image"]
    # desc = data["language_instruction"]
    # true_actions = data["actions"]
    # pred_actions = model.inference(rgb, desc, diffusion_iters=5)

    # print(f"Error: {np.mean((true_actions - pred_actions)**2)}")
