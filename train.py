
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import copy
import matplotlib.pyplot as plt
import numpy as np


from dataset import FluidDataset
from model import UNet_conditional, EMA
from diffusion import Diffusion



TRAIN_MEAN = 0.255825  
TRAIN_STD = 0.659223


BASE_DIR = "/root/lanyun-tmp/code/"
LIFT_PATH = os.path.join(BASE_DIR, "train_shengli.txt")
DRAG_PATH = os.path.join(BASE_DIR, "train_zuli.txt")
NPY_DIR = os.path.join(BASE_DIR, "train_npy_180/")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints/")
VIS_DIR = os.path.join(BASE_DIR, "training_vis/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 2000
IMG_SIZE = 128
EMA_BETA = 0.995

VIS_INTERVAL = 10 

def save_visualization(ema_model, diffusion, cond, gt_img, epoch):
    """

    """
    ema_model.eval()
    with torch.no_grad():
        # n=1, conditions=cond
        sampled_img = diffusion.sample(ema_model, n=1, conditions=cond)
    
    gen_np = sampled_img[0, 0].cpu().numpy() * TRAIN_STD + TRAIN_MEAN
    gt_np = gt_img[0, 0].cpu().numpy() * TRAIN_STD + TRAIN_MEAN
    
    plt.figure(figsize=(10, 5))
    
    # Ground Truth
    plt.subplot(1, 2, 1)
    plt.title(f"Ground Truth (Epoch {epoch})")
    plt.imshow(gt_np, cmap='jet', origin='lower')
    plt.colorbar(label='u_x')
    plt.axis('off')
    
    # Generated
    plt.subplot(1, 2, 2)
    plt.title(f"Generated (EMA)")
    plt.imshow(gen_np, cmap='jet', origin='lower')
    plt.colorbar(label='u_x')
    plt.axis('off')
    

    save_path = os.path.join(VIS_DIR, f"vis_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Visualization saved to {save_path}")

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True) 
    logging.info(f"Using device: {DEVICE}")
    

    dataset = FluidDataset(
        lift_file=LIFT_PATH,
        drag_file=DRAG_PATH,
        npy_dir=NPY_DIR,
        target_size=IMG_SIZE,
        mean=TRAIN_MEAN,
        std=TRAIN_STD
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    logging.info(f"Dataset loaded. Size: {len(dataset)}")

    viz_iter = iter(dataloader)
    viz_cond_batch, viz_img_batch = next(viz_iter)
    
    fixed_viz_cond = viz_cond_batch[0:1].to(DEVICE) # shape [1, 3]
    fixed_viz_gt = viz_img_batch[0:1].to(DEVICE)    # shape [1, 1, 128, 128]
    
    print(f"Visualization sample fixed. Cond: {fixed_viz_cond.cpu().numpy()}")
    # ==========================================================


    model = UNet_conditional(c_in=1, c_out=1, device=DEVICE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=IMG_SIZE, device=DEVICE)


    ema = EMA(EMA_BETA)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(DEVICE)

    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch+1}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        
        for i, (cond, images) in enumerate(pbar):
            images = images.to(DEVICE)
            cond = cond.to(DEVICE)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t, cond)
            loss = mse(noise, predicted_noise)

            if i == 0 and epoch % 1 == 0:
                print(f"\n[Debug] Pred Mean: {predicted_noise.mean().item():.4f}, Std: {predicted_noise.std().item():.4f}")
                print(f"[Debug] GT Noise Mean: {noise.mean().item():.4f}, Std: {noise.std().item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ema.step_ema(ema_model, model)
            
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        

        if (epoch + 1) % VIS_INTERVAL == 0:
            logging.info("Running visualization...")
            save_visualization(ema_model, diffusion, fixed_viz_cond, fixed_viz_gt, epoch + 1)


        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch+1}.pt"))
            torch.save(ema_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ema_ckpt_epoch_{epoch+1}.pt"))
            print(f"Epoch {epoch+1} Saved. (Checkpoints & EMA)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()