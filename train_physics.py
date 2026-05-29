import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from jiadataset_physics import FluidPhysicsDataset
from model_physics import UNet_conditional_1D, EMA
from diffusion import Diffusion

BASE_DIR = "/work/home/self/flow/JFM_major/1/dataset/"


FEATURE_FILES = {
    "lift": os.path.join(BASE_DIR, "l-Copy1.txt"),
    "drag": os.path.join(BASE_DIR, "d-Copy1.txt"),
    "c1": os.path.join(BASE_DIR, "z1-Copy1.txt"),
    "c2": os.path.join(BASE_DIR, "z2-Copy1.txt"),
    "c3": os.path.join(BASE_DIR, "z3-Copy1.txt"),
    "c4": os.path.join(BASE_DIR, "z4-Copy1.txt"),
    "c5": os.path.join(BASE_DIR, "z5-Copy1.txt"),
    "c6": os.path.join(BASE_DIR, "z6-Copy1.txt"),
    "c7": os.path.join(BASE_DIR, "z7-Copy1.txt"),
    "c8": os.path.join(BASE_DIR, "z8-Copy1.txt"),
    "c9": os.path.join(BASE_DIR, "z9-Copy1.txt"),
    "c10": os.path.join(BASE_DIR, "z10-Copy1.txt"),
    "c11": os.path.join(BASE_DIR, "z11-Copy1.txt"),
    "c12": os.path.join(BASE_DIR, "z12-Copy1.txt"),
    "c13": os.path.join(BASE_DIR, "z13-Copy1.txt"),
    "c14": os.path.join(BASE_DIR, "z14-Copy1.txt"),
    "c15": os.path.join(BASE_DIR, "z15-Copy1.txt"),
    "c16": os.path.join(BASE_DIR, "z16-Copy1.txt"),

}

NPY_DIR = os.path.join(BASE_DIR, "trainnpy/")
STATS_PATH = os.path.join(BASE_DIR, "qian200norm_stats_256x64_5channel.npz")

CHECKPOINT_DIR = "./checkpoints_physics/"
VIS_DIR = "./vis_physics/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 10000
VIS_INTERVAL = 50


NU = 1.0 / 100.0

DX = 20.0 / 256.0
DY = 6.0 / 64.0



def save_visualization(ema_model, diffusion, cond, gt_img, epoch, mean_np, std_np):
    ema_model.eval()
    B, C, H, W = gt_img.shape

    with torch.no_grad():
        x = torch.randn((B, C, H, W)).to(DEVICE)
        for i in tqdm(reversed(range(diffusion.noise_steps)), desc="vis", leave=False):
            t = (torch.ones(B) * i).long().to(DEVICE)
            predicted_noise = ema_model(x, t, cond)

            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]

            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
            ) + torch.sqrt(beta) * noise

        sampled_img = x

    gen_np = sampled_img[0].cpu().numpy() * std_np[:, None, None] + mean_np[:, None, None]
    gt_np = gt_img[0].cpu().numpy() * std_np[:, None, None] + mean_np[:, None, None]

    field_names = ['u-velocity', 'v-velocity', 'Pressure']
    cmap_names = ['jet', 'jet', 'viridis']

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
    fig.suptitle(f"Epoch {epoch} - Full Transient Physics", fontsize=16)

    for c in range(3):
        vmin = min(gt_np[c].min(), gen_np[c].min())
        vmax = max(gt_np[c].max(), gen_np[c].max())

        ax_gt = axes[c, 0]
        im_gt = ax_gt.imshow(gt_np[c], cmap=cmap_names[c], origin='lower', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"GT: {field_names[c]}")
        ax_gt.axis('off')
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_gen = axes[c, 1]
        im_gen = ax_gen.imshow(gen_np[c], cmap=cmap_names[c], origin='lower', vmin=vmin, vmax=vmax)
        ax_gen.set_title(f"Generated: {field_names[c]}")
        ax_gen.axis('off')
        fig.colorbar(im_gen, ax=ax_gen, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, f"vis_phys_ep_{epoch}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    ema_model.train()


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    dataset = FluidPhysicsDataset(
        feature_files=FEATURE_FILES,
        npy_dir=NPY_DIR,
        stats_path=STATS_PATH
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    mean_tensor = torch.tensor(dataset.mean_img).view(1, 5, 1, 1).to(DEVICE)
    std_tensor = torch.tensor(dataset.std_img).view(1, 5, 1, 1).to(DEVICE)
    mean_np = dataset.mean_img
    std_np = dataset.std_img


    dynamic_cond_dim = dataset.cond_dim
    print(f"Dataset loaded. Size: {len(dataset)}")
    print(f"Condition dimension: {dynamic_cond_dim}")

    viz_iter = iter(dataloader)
    fixed_cond_batch, fixed_gt_batch = next(viz_iter)
    fixed_cond = fixed_cond_batch[0:1].to(DEVICE)
    fixed_gt = fixed_gt_batch[0:1].to(DEVICE)

    print(f"Fixed cond shape: {fixed_cond.shape}")
    print(f"Fixed gt shape: {fixed_gt.shape}")


    model = UNet_conditional_1D(
        c_in=5,
        c_out=5,
        cond_in=dynamic_cond_dim,
        device=DEVICE
    ).to(DEVICE)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(DEVICE)
    ema = EMA(beta=0.995)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=None, device=DEVICE)

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for cond, images in pbar:
            images = images.to(DEVICE)
            cond = cond.to(DEVICE)

            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t, cond)


            loss_data = mse(noise, predicted_noise)


            alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None].to(DEVICE)
            x0_hat = (x_t - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)

            x0_phys = x0_hat * std_tensor + mean_tensor
            u, v, p, u_t, v_t = (
                x0_phys[:, 0:1],
                x0_phys[:, 1:2],
                x0_phys[:, 2:3],
                x0_phys[:, 3:4],
                x0_phys[:, 4:5],
            )


            du_dy, du_dx = torch.gradient(u, spacing=(DY, DX), dim=(2, 3))
            dv_dy, dv_dx = torch.gradient(v, spacing=(DY, DX), dim=(2, 3))
            dp_dy, dp_dx = torch.gradient(p, spacing=(DY, DX), dim=(2, 3))

            u_yy = torch.gradient(du_dy, spacing=(DY, DX), dim=(2, 3))[0]
            u_xx = torch.gradient(du_dx, spacing=(DY, DX), dim=(2, 3))[1]
            v_yy = torch.gradient(dv_dy, spacing=(DY, DX), dim=(2, 3))[0]
            v_xx = torch.gradient(dv_dx, spacing=(DY, DX), dim=(2, 3))[1]


            div = du_dx + dv_dy
            loss_cont = (div ** 2).mean()


            mom_u = u_t + (u * du_dx) + (v * du_dy) + dp_dx - NU * (u_xx + u_yy)
            mom_v = v_t + (u * dv_dx) + (v * dv_dy) + dp_dy - NU * (v_xx + v_yy)
            loss_mom = (mom_u ** 2).mean() + (mom_v ** 2).mean()




            loss = loss_data + loss_cont + loss_mom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            scaled_phys_loss = current_lambda * (loss_cont + loss_mom).item()
            pbar.set_postfix(
                Data=f"{loss_data.item():.4f}",
                Phys=f"{scaled_phys_loss:.6f}",
                CondDim=dynamic_cond_dim
            )

        if (epoch + 1) % VIS_INTERVAL == 0:
            save_visualization(
                ema_model, diffusion, fixed_cond, fixed_gt, epoch + 1, mean_np, std_np
            )

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_phys_ep_{epoch+1}.pt"))
            torch.save(ema_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ema_phys_ep_{epoch+1}.pt"))


if __name__ == "__main__":
    train()