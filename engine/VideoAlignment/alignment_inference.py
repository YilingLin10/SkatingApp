import tensorflow as tf
import os
import numpy as np
from dtw import *

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import gc
# flags
from absl import flags
from absl import app
# utils
from model.embedder import Embedder
from config import CONFIG
import data.utils as tccdata
import data.skeleton as tccskeleton

# Specify GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

###### USAGE #####
# predictor = AlignmentPredictor()
# predictor.predict()
flags.DEFINE_string('mode', None, 'the dir where the 2 videos to align locate')
FLAGS = flags.FLAGS

class AlignmentPredictor():
    def __init__(self, mode):
        super().__init__()
        self.ckpt_path = "/home/lin10/projects/SkatingApp/engine/VideoAlignment/checkpoint/flip/"
        self.PATH_TO_RAW_VIDEOS = "/home/lin10/projects/SkatingApp/cache/align/data"
        self.dataset = "flip"
        self.mode = mode
        self.output_dir = "/home/lin10/projects/SkatingApp/results"
        self.model = self.__get_model() 
        
    def __get_model(self):
        model = Embedder(CONFIG.EMBEDDING_SIZE, CONFIG.NORMALIZE_EMBEDDINGS, CONFIG.NUM_CONTEXT_STEPS)
        optimizer = tf.keras.optimizers.Adam(CONFIG.LEARNING_RATE) 
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        return model
    
    def get_embs(self, videos, video_seq_lens, frames_per_batch,
                num_context_steps, context_stride):
        tf.keras.backend.set_learning_phase(0)
        embs_list = []
        for video, seq_len in zip(videos, video_seq_lens):
            embs = []
            num_batches = int(np.ceil(float(seq_len)/frames_per_batch))
            for i in range(num_batches):
                steps = np.arange(i*frames_per_batch, (i+1)*frames_per_batch)
                steps = np.clip(steps, 0, seq_len-1)
                def get_context_steps(step):
                    return tf.clip_by_value(
                    tf.range(step - (num_context_steps - 1) * context_stride,
                            step + context_stride,
                            context_stride),
                            0, seq_len-1)
                steps_with_context = tf.reshape(
                    tf.map_fn(get_context_steps, steps), [-1])
                frames = tf.gather(video, steps_with_context)
                frames = tf.cast(frames, tf.float32)
                frames = (frames/127.5)-1.0
                frames = tf.image.resize(frames, (168, 168)) 
                frames = tf.expand_dims(frames, 0) 
                embs.extend(self.model(frames, training=False).numpy()[0])
            embs = embs[:seq_len]
            assert len(embs) == seq_len
            embs = np.asarray(embs)
            embs_list.append(embs)
        return embs_list
    
    def visualize(self, start_frames, frames, query=0, candi=1):
        OUTPUT_PATH = '{}/align_{}.mp4'.format(self.output_dir, self.mode)
        
        # Create subplots
        nrows = len(frames)
        fig, ax = plt.subplots(
                ncols=nrows,
                figsize=(10 * nrows, 10 * nrows),
                tight_layout=True)
        
        def unnorm(query_frame):
            min_v = query_frame.min()
            max_v = query_frame.max()
            query_frame = (query_frame - min_v) / (max_v - min_v)
            return query_frame

        ims = []
        def init():
            k = 0
            for k in range(nrows):
                ims.append(ax[k].imshow(
                    unnorm(frames[k][0])))
                ax[k].grid(False)
                ax[k].set_xticks([])
                ax[k].set_yticks([])
            return ims

        ims = init()

        # The one with larger start_frame needs to be played first
        first = query if (start_frames[query] > start_frames[candi]) else candi
        second = candi if (first==query) else query
        # The second video starts playing at start_frame
        start_frame = start_frames[first] - start_frames[second]
        if (start_frames[first] + len(frames[second])) > len(frames[first]):
            num_total_frames = start_frames[first] + len(frames[second])
        else:
            num_total_frames = len(frames[first])

        def update(i):
            if i < len(frames[first]):
                ims[first].set_data(unnorm(frames[first][i]))
            else:
                ims[first].set_data(unnorm(frames[first][-1]))
            ax[first].set_title('FRAME {}'.format(i), fontsize = 14)

            if i >= start_frame and i < (start_frame+len(frames[second])):
                ims[second].set_data(unnorm(frames[second][i-start_frame]))
            elif i < start_frame:
                ims[second].set_data(unnorm(frames[second][0]))
            else:
                ims[second].set_data(unnorm(frames[second][-1]))
            plt.tight_layout()
            return ims

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(num_total_frames),
            interval=100,
            blit=False)
        anim.save(OUTPUT_PATH, dpi=40)

        plt.close()
        
    def predict(self, query=0, candi=1):
        videos, video_seq_lens, videos_raw, skeletons = tccdata.load_skate_data(self.PATH_TO_RAW_VIDEOS, self.dataset, self.mode)
        if videos is None:
            return None
        standard_vid, standard_seq_len, standard_raw, standard_skeleton = tccdata.load_skate_data(self.PATH_TO_RAW_VIDEOS, self.dataset, 'standard')
        print('------------------------------------------------------')
        print('-----------------Finish loading data.-----------------')
        print('------------------------------------------------------')
        
        print("Extracting per-frame embeddings...")
        embs = self.get_embs(videos, video_seq_lens,
                        frames_per_batch=CONFIG.FRAMES_PER_BATCH, 
                        num_context_steps=CONFIG.NUM_CONTEXT_STEPS,
                        context_stride=CONFIG.CONTEXT_STRIDE)
        standard_emb = self.get_embs(standard_vid, standard_seq_len,
                        frames_per_batch=CONFIG.FRAMES_PER_BATCH, 
                        num_context_steps=CONFIG.NUM_CONTEXT_STEPS,
                        context_stride=CONFIG.CONTEXT_STRIDE)
        
        def dist_fn(x, y):
            dist = np.sum((x-y)**2)
            return dist
        
        def get_start_frame(emb, emb_name, standard_emb):
            min_dists = []
            for i in range(len(emb)-len(standard_emb)):
                query_embs = emb[i:i+len(standard_emb)]
                candidate_embs = standard_emb
                min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_embs, candidate_embs, dist=dist_fn)
                min_dists.append(min_dist)
            start_frame = min_dists.index(min(min_dists))
            return start_frame
        
        start_frames = [0, 0]
        start_frames[query] = get_start_frame(embs[query], 'query', standard_emb[0])
        start_frames[candi] = get_start_frame(embs[candi], 'candi', standard_emb[0])
        # Draw skeleton on raw videos
        videos_drawn = tccskeleton.vis_skeleton(videos_raw, skeletons)
        self.visualize(start_frames, videos_drawn)
        
def main(_argv):
    predictor = AlignmentPredictor(mode=FLAGS.mode)
    predictor.predict()
    gc.collect()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass