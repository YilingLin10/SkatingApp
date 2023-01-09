from __future__ import division
from __future__ import print_function

from model.stgcn_encoder_crf import STGCN_Transformer
from data.dataset import IceSkatingDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
from absl import flags
from absl import app
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import cv2
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

flags.DEFINE_string('video_name', None, 'the name of the test video')
FLAGS = flags.FLAGS

#### USAGE ###
#     predictor = Predictor()
#     jump_length_list = predictor.predict(video_name)

class Predictor():
    def __init__(self):
        super().__init__()
        self.model_type = "STGCN_ENCODER_CRF"
        self.ckpt_path = "/home/lin10/projects/SkatingApp/engine/JumpDetection/checkpoint/stgcn_encoder_crf/save/transformer_bin_class.pth"
        self.alphapose_root_dir = "/home/lin10/projects/SkatingApp/cache/jump/data/alphapose"
        self.pkl_root_dir = "/home/lin10/projects/SkatingApp/cache/jump/data/pkl"
        self.output_dir = "/home/lin10/projects/SkatingApp/results"
        self.tag2idx = {
                        "O": 0,
                        "B": 1, 
                        "I": 2,
                        "E": 3
                    }
        self.device = 'cuda'
        self.model = self.__get_model()
    
    def __get_model(self):
        model = STGCN_Transformer(
            hidden_channel = 32,
            out_channel = 128,
            nhead = 4, 
            num_encoder_layers = 2,
            dim_feedforward = 256,
            dropout = 0.1,
            batch_first = True,
            num_class = 4,
            use_crf = True
        ).to(self.device)
        model.eval()
        ckpt = torch.load(self.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        return model
    
    def same_seed(self, seed): 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def compute_length_in_seconds(self, prediction, fps):
        def frames_to_seconds(num_frames):
            return float(num_frames) / fps 
        jump_frames = [i for i, p in enumerate(prediction) if p != 0]
        print(jump_frames)
        length_list = []
        i = 0
        l = 0
        while i < len(jump_frames):
            l += 1
            # last one
            if i == len(jump_frames) -1:
                length_list.append(l)
                break
            # discontinuity
            if jump_frames[i] + 1 != jump_frames[i+1]:
                length_list.append(l)
                l = 0
            i += 1
        if len(length_list) == 0:
            length_list = [0]
            
        length_list = [frames_to_seconds(length) for length in length_list]
        return length_list
    
    def visualize(self, video_name, prediction):
        def get_frames():
            frames_dir = f"{self.alphapose_root_dir}/alpha_pose_{video_name}/vis"
            files = os.listdir(frames_dir)
            imgs = [f for f in files if '.jpg' in f]
            for img in imgs:
                new_name = '{0:04d}'.format(int(img.rstrip('.jpg')))+'.jpg'
                old = os.path.join(frames_dir, img)
                new = os.path.join(frames_dir, new_name)
                os.rename(old, new)

            # Preprocessing raw frames
            framefiles = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
            frames_raw = []
            for framefile in framefiles:
                frame_raw = cv2.imread(framefile)
                frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
                frames_raw.append(frame_raw)
                
            return frames_raw

        def unnorm(frame):
            min_v = frame.min()
            max_v = frame.max()
            frame = (frame - min_v) / (max_v - min_v)
            return frame
        
        def gray_scale(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray
        
        frames = get_frames()
        OUTPUT_PATH = '{}/jump_{}.mp4'.format(self.output_dir, video_name)
        jumps = [i for i, label in enumerate(prediction) if label != 0]
        fig = plt.figure(figsize=(11.25, 20),tight_layout=True)
        im = plt.imshow(unnorm(frames[0]), 'gray')
        plt.axis('off')
        
        def update(i):
            if i in jumps:
                im.set_data(unnorm(frames[i]))
            else:
                im.set_data(unnorm(gray_scale(frames[i])))

            # fig.suptitle('FRAME {}'.format(i), fontsize = 36, y=0.95)
            return im

        print("Generating animation.....")
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(frames)),
            interval=200,
            blit=False)
        anim.save(OUTPUT_PATH, dpi=40)
        plt.close()
    
    def get_dataloader(self, video_name):
        pkl_path = '{}/{}.pkl'.format(self.pkl_root_dir, video_name)
        dataset = IceSkatingDataset(pkl_file=pkl_path,
                                subtract_feature=False)
        dataloader = DataLoader(dataset,batch_size=1,
                            shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
        return dataloader
            
    def predict(self, video_name):
        self.same_seed(42)
        dataloader = self.get_dataloader(video_name)
        
        with open(f'{self.alphapose_root_dir}/alpha_pose_{video_name}/info.json', 'r') as f:
            video_info = json.load(f)
            
        for batch in dataloader:
            with torch.no_grad():
                if (video_info['width'] < 1000 or video_info['height'] < 1000):
                    keypoints = torch.mul(batch['keypoints'], 2)
                else:
                    keypoints = batch['keypoints']    

                keypoints, mask = keypoints.to(self.device), batch['mask'].to(self.device)
                preds = self.model(keypoints, mask)
        
        self.visualize(video_name, preds[0])
        
        jump_length_list = self.compute_length_in_seconds(preds[0], video_info['fps'])
        with open(f'{self.output_dir}/jump_{video_name}.txt', 'w') as f:
            for l in jump_length_list:
                f.write(f"{round(l,2)}\n")
    

def main(_argv):
    predictor = Predictor()
    predictor.predict(FLAGS.video_name)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass