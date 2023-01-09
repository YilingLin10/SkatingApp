import numpy as np
import os 
import glob
import json
import cv2
from scipy.spatial import distance
from pathlib import Path
import csv
from absl import flags
from absl import app
import pickle
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
flags.DEFINE_string('video_name', None, 'the name of the test video')
FLAGS = flags.FLAGS

class Preprocesser():
    def __init__(self, video_name):
        super().__init__()
        self.video_name = video_name
        self.root_dir = "/home/lin10/projects/SkatingApp/cache/jump/data/alphapose/alpha_pose_{}".format(video_name)
        self.output_file = "/home/lin10/projects/SkatingApp/cache/jump/data/pkl/{}.pkl".format(video_name)
        self.sample_list = []
        self.SKELETON_CONF_THRESHOLD = 0.0
    
    def writePickle(self):
        with open(self.output_file, "wb") as f:
            pickle.dump(self.sample_list, f)
    
    def get_main_skeleton(self):
        path_to_json = "{}/alphapose-results.json".format(self.root_dir)
        root_dir = self.root_dir
        frames = list(Path(os.path.join(self.root_dir, 'vis')).glob("*.jpg"))
        last_frame = sorted([int(os.path.split(frame)[1].replace('.jpg','')) for frame in frames])[-1]
        with open(path_to_json) as f:
            json_skeleton = json.load(f)
        
        # Get first image id
        prev_frame = int(json_skeleton[0]['image_id'].rstrip('.jpg'))
        score_list = []
        main_skeleton_list = []
        for skeleton in json_skeleton:
            # Get current image id
            cur_frame = int(skeleton['image_id'].rstrip('.jpg'))
            # Get all skeleton in current frame
            if cur_frame == prev_frame:
                score_list.append([skeleton['score'], np.reshape(skeleton['keypoints'], (17,3))])
            # Get the main skeleton in previous frame by score or distance
            else:
                if main_skeleton_list == []:
                    main_skeleton_list.append([prev_frame, np.array(max(score_list))])
                else:
                    dist_list = []
                    # Pick a skeleton that is the closest to the previous one
                    for score in score_list:
                        ske_distance = distance.euclidean(np.delete(score[1], (2), axis=1).ravel(), np.delete(main_skeleton_list[-1][1][1], (2), axis=1).ravel())
                        dist_list.append([ske_distance, score])
                    main_skeleton_list.append([prev_frame, np.array(min(dist_list)[1])])
                # Clear score list and append first skeleton for current frame
                score_list.clear()
                score_list.append([skeleton['score'], np.reshape(skeleton['keypoints'], (17,3))])
            prev_frame = cur_frame
        # append max score for the last frame
        main_skeleton_list.append([cur_frame, np.array(max(score_list))])
        main_skeleton_list = np.reshape(main_skeleton_list, (-1,2))

        keypoints_info = []
        for main_skeleton in main_skeleton_list:
            delete_pt = []
            keypoints = main_skeleton[1][1]
            for row in range(len(keypoints)):
                if keypoints[row][2] < self.SKELETON_CONF_THRESHOLD:
                    delete_pt.append(row)
            keypoints = np.delete(keypoints, (delete_pt), axis=0)
            keypoints_info.append([main_skeleton[0], keypoints])
        keypoints_info = np.array(keypoints_info)

        # Insert missing frame info
        for i in range(last_frame+1):
            if i not in keypoints_info[:,0]:
                insert_row = np.concatenate([[i],keypoints_info[i-1,1:]], axis=0)
                keypoints_info = np.insert(keypoints_info, i, insert_row, axis=0)
        return keypoints_info
    
    def subtract_features(self, keypoints):
        subtracted_keypoints = []
        for i in range(1, 17, 2):
            subtraction = keypoints[i][0] - keypoints[i+1][0]
            subtracted_keypoints.append(subtraction)
        return np.array(subtracted_keypoints)
    
    def load_data(self):
        frames = list(Path(os.path.join(self.root_dir, 'vis')).glob("*.jpg"))
        alphaposeResults = self.get_main_skeleton()
        assert len(frames) == len(alphaposeResults), "{} error".format(self.video_name)
        subtractions_list = [self.subtract_features(alphaposeResult[1]) for alphaposeResult in alphaposeResults]
        keypoints_list = [np.delete(alphaposeResult[1], 2, axis=1).reshape(-1) for alphaposeResult in alphaposeResults]
        subtraction_features_list = np.append(keypoints_list, subtractions_list, axis=1)  
        features_list = keypoints_list
        sample = {"subtraction_features": subtraction_features_list, "features": features_list, "video_name": self.video_name}
        self.sample_list.append(sample)   
        self.writePickle()

def main(_argv):
    preprocesser = Preprocesser(FLAGS.video_name)
    preprocesser.load_data()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass