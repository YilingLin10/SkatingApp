import numpy as np
import os 
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
from pathlib import Path
import pickle
import json

class IceSkatingDataset(Dataset):

    def __init__(self, pkl_file, subtract_feature):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(pkl_file, 'rb') as f:
            self.video_data_list = pickle.load(f)
        self.subtract_feature = subtract_feature
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.video_data_list[idx]['video_name']

        if self.subtract_feature:
            features_list = self.video_data_list[idx]['subtraction_features']
        else:
            features_list = self.video_data_list[idx]['features']
            
        mask = np.ones((1, len(features_list)))
        mask = torch.tensor(mask).bool()
        keypoints = np.array(features_list)
        sample = {"keypoints": features_list, "video_name": video_name, 'mask': mask}
        return sample
    
    def collate_fn(self, samples):
        d_model = samples[0]['keypoints'][0].shape[0]
        keypoints = np.zeros((len(samples), len(samples[0]['keypoints']), d_model))
        keypoints[0] = samples[0]['keypoints']
        keypoints = torch.FloatTensor(keypoints)
        return {"keypoints": keypoints, "mask": samples[0]['mask']}

if __name__ == '__main__':
    dataset = IceSkatingDataset(pkl_file='/home/lin10/projects/SkatingApp/cache/jump/data/pkl/response.pkl',
                                subtract_feature=False)

    dataloader = DataLoader(dataset,batch_size=1,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        # print(sample_batched['mask'])
        print(sample_batched['keypoints'])