import os
import torch
from torch.utils.data import Dataset
from utils import extract_mouth_frames
class CremadLipreadingDataset(Dataset):
    def __init__(self,sample_list,task):
        self.samples=sample_list
        self.task=task
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        video_path,transcript=self.samples[idx]
        frames=extract_mouth_frames(video_path)
        frames_tensor=torch.from_numpy(frames).unsqueeze(1).float()
        bos=self.task.target_dictionary.bos()
        eos=self.task.target_dictionary.eos()
        txt=transcript.strip()
        tokens=[self.task.target_dictionary.index(c) for c in list(txt.lower()) if c!=" "]
        target=[bos]+tokens+[eos]
        target_tensor=torch.tensor(target,dtype=torch.long)
        return frames_tensor,target_tensor

