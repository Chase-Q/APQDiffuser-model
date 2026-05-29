import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class FluidPhysicsDataset(Dataset):
    def __init__(self, feature_files, npy_dir, stats_path):

        self.npy_dir = npy_dir

        stats = np.load(stats_path)
        self.mean_img = stats['mean'].astype(np.float32)
        self.std_img = stats['std'].astype(np.float32)

        self.feature_names = list(feature_files.keys())
        self.meta = None

        for feat_name, file_path in feature_files.items():

            df = pd.read_csv(
                file_path,
                delim_whitespace=True,
                header=None,
                names=["sim_id", feat_name, "time"]
            )

            if self.meta is None:

                self.meta = df.copy()
            else:

                self.meta = pd.merge(self.meta, df[["sim_id", feat_name]], on="sim_id")


        self.meta = self.meta.sort_values(by="sim_id").reset_index(drop=True)

        self.cond_columns = self.feature_names + ["time"]

        cond_features = self.meta[self.cond_columns].values.astype(np.float32)
        self.cond_mean = np.mean(cond_features, axis=0)
        self.cond_std = np.std(cond_features, axis=0)
        self.cond_std = np.where(self.cond_std == 0, 1e-8, self.cond_std)

    @property
    def cond_dim(self):
        return len(self.cond_columns)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]


        raw_cond = row[self.cond_columns].values.astype(np.float32)
        cond_tensor = torch.from_numpy((raw_cond - self.cond_mean) / self.cond_std)


        filepath = os.path.join(self.npy_dir, f"200z-{int(row['sim_id'])}.npy")
        try:
            img_array = np.load(filepath).astype(np.float32)
        except FileNotFoundError:
            img_array = np.zeros((5, 64, 256), dtype=np.float32)


        img_array = (img_array - self.mean_img[:, None, None]) / self.std_img[:, None, None]
        img_tensor = torch.from_numpy(img_array)

        return cond_tensor, img_tensor