# sample.py
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import pandas as pd
from model import DiffusionModel
from torch.utils.data import DataLoader
from dataset_val import ForceToXVelocity2DDataset  # 用于加载条件向量和预处理

def smooth_curve(curve, window_size=11):
    import numpy as np
    import scipy.ndimage

    smoothed = scipy.ndimage.uniform_filter1d(curve, size=window_size, axis=-1)
    return smoothed

def plot_curve(curve: torch.Tensor, save_path="curve_plot.png"):
    """
    curve: Tensor of shape [7, L]
    curve[0] is x-axis, curve[1:] are y curves
    """
    curve = curve.cpu().numpy()
    x = curve[0]  # x-axis: [L]
    curve = smooth_curve(curve, window_size=11)
    plt.figure(figsize=(10, 6))

    #for i in range(1, 7):
    for i in range(1, 2):
        y = curve[i]
        plt.plot(x, y, label=f'channel_{i}')

    plt.xlabel("X (curve[0])")
    plt.ylabel("Value")
    plt.title("Generated Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved curve plot to {save_path}")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'samples'
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 1
val_dataset = ForceToXVelocity2DDataset()
dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = DiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("ckpts_v4/model_epoch10.pt", map_location=DEVICE))
model.eval()


scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)
scheduler.set_timesteps(50)


for cond, image in tqdm(dataloader):
    cond = cond.to(DEVICE)
    break

 
import torchvision.transforms as T
from PIL import Image
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])
 

B = conditions.size(0)
image = torch.randn(B, 1, 176, 176).to(DEVICE)



for t in scheduler.timesteps:

    t = t.to(DEVICE)
    with torch.no_grad():
        eps_image = model(image, t, conditions)
    
    image = scheduler.step(eps_image, t, image).prev_sample



