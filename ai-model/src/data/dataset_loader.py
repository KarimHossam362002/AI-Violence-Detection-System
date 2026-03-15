# src/data/dataset_loader.py

import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset

class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=16):
        self.samples = []
        self.seq_len = seq_len

        for label, folder in enumerate(['normal', 'violence']):
            path = os.path.join(root_dir, folder)
            for v in os.listdir(path):
                self.samples.append((os.path.join(path, v), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, label = self.samples[idx]
        frames = self._read(video)
        frames = torch.tensor(frames).permute(0,3,1,2)
        return frames.float(), torch.tensor(label)

    def _read(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < self.seq_len:
            ret, f = cap.read()
            if not ret: break
            f = cv2.resize(f, (224,224)) / 255.0
            frames.append(f)

        cap.release()
        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        return np.array(frames)
