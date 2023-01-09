import os
import json

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from matplotlib.pyplot import draw
import numpy as np
import cv2
from scipy.spatial import distance
import math

# Return bbox [image_id, middle_x, middle_y, width, height]
def get_bbox(path_to_json):
  SKELETON_CONF_THRESHOLD = 0.0
  try:
    with open(os.path.join(path_to_json, 'alphapose-results.json')) as f:
      json_skeleton = json.load(f)
  except FileNotFoundError as not_found:
    print(f"Alphapose failed. {not_found.filename} alphapose-results.json not found.")
    return
  
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
        # Pick the skeleton which is close to previos one
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

  # Build skeleton info [image_id, middle_x, middle_y, width, height]
  bbox_info = []
  for main_skeleton in main_skeleton_list:
    delete_pt = []
    keypoints = main_skeleton[1][1]
    for row in range(len(keypoints)):
      if keypoints[row][2] < SKELETON_CONF_THRESHOLD:
        delete_pt.append(row)
    keypoints = np.delete(keypoints, (delete_pt), axis=0)
    middle_pt = np.mean(np.delete(keypoints, (2), axis=1), axis=0)
    skeleton_width = max(keypoints[:,0])-min(keypoints[:,0])
    skeleton_height = max(keypoints[:,1])-min(keypoints[:,1])
    bbox_info.append([main_skeleton[0], middle_pt[0], middle_pt[1], skeleton_width, skeleton_height])
  bbox_info = np.array(bbox_info)

  # Insert missing frame info
  for i in range(cur_frame+1):
    if i not in bbox_info[:,0]:
      insert_row = np.concatenate([[i],bbox_info[i-1,1:]], axis=0)
      bbox_info = np.insert(bbox_info, i, insert_row, axis=0)

  return bbox_info

# Return keypoint data for each frame [image_id, array(17, 3)]
def get_main_skeleton(path_to_json):
  SKELETON_CONF_THRESHOLD = 0.0
  with open(os.path.join(path_to_json, 'alphapose-results.json')) as f:
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
        # Pick the skeleton which is close to previos one
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
      if keypoints[row][2] < SKELETON_CONF_THRESHOLD:
        delete_pt.append(row)
    keypoints = np.delete(keypoints, (delete_pt), axis=0)
    keypoints_info.append([main_skeleton[0], keypoints])
  keypoints_info = np.array(keypoints_info)

  # Insert missing frame info
  for i in range(cur_frame+1):
    if i not in keypoints_info[:,0]:
      insert_row = np.concatenate([[i],keypoints_info[i-1,1:]], axis=0)
      keypoints_info = np.insert(keypoints_info, i, insert_row, axis=0)

  return keypoints_info

# Draw skeleton data for one video
def vis_skeleton(videos, video_keypoints, vis_thres=0.4):
    l_pair = [
      (0, 1), (0, 2), (1, 3), (2, 4),  # Head
      (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
      (5, 11), (6, 12), (5, 6), (11, 12), # Body
      (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
              (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
              (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle
    videos_drawn = []
    for video, keypoints in zip(videos, video_keypoints):
      part_line = {}
      drawn_frames = []
      for frame, keypoint in zip(video, keypoints):
        # Draw keypoints
        invis_pt = []
        for n, pt in zip(range(len(keypoint[1])), keypoint[1]):
          cor_x, cor_y = int(pt[0]), int(pt[1])
          part_line[n] = (int(cor_x), int(cor_y))
          if pt[2]>vis_thres:
            frame = cv2.circle(frame, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
          else:
            invis_pt.append(n)
        # Draw limbs
        for (start_p, end_p) in l_pair:
          if start_p in part_line and end_p in part_line and start_p not in invis_pt and end_p not in invis_pt:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            frame = cv2.line(frame, start_xy, end_xy, (255,255,255), 1)
        drawn_frames.append(frame)
      drawn_frames = np.asarray(drawn_frames)
      videos_drawn.append(drawn_frames)
    return videos_drawn