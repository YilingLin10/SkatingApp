import cv2
import glob
import os
import numpy as np
import tensorflow as tf

from .skeleton import get_bbox, get_main_skeleton

def pad_zeros(frames, max_seq_len):
  npad = ((0, max_seq_len-len(frames)), (0, 0), (0, 0), (0, 0))
  frames = np.pad(frames, pad_width=npad, mode='constant', constant_values=0)
  return frames

def load_skate_data(path_to_raw_videos, dataset, mode):
  width = 224
  height = 224
  folder = os.path.join(path_to_raw_videos, dataset, mode)
  video_dirnames = sorted(os.listdir(folder))
  print('Found %d videos to align.'%len(video_dirnames))

  # Rename frame files for further sorting
  try:
    for video_dir in video_dirnames:
      imgs = os.listdir(os.path.join(folder, video_dir, 'vis'))
      for img in imgs:
        new_name = '{0:04d}'.format(int(img.rstrip('.jpg')))+'.jpg'
        old = os.path.join(folder, video_dir, 'vis', img)
        new = os.path.join(folder, video_dir, 'vis', new_name)
        os.rename(old, new)
  except FileNotFoundError as not_found:
    print(f"Alphapose failed. {not_found.filename} directory not found.")
    return None, None, None, None
    
  # Preprocessing raw frames
  videos_raw = []
  videos = []
  video_seq_lens = []
  skeletons = []
  for video_dir in video_dirnames:
    bboxes = get_bbox(os.path.join(folder, video_dir))
    skeleton = get_main_skeleton(os.path.join(folder, video_dir))
    framefiles = sorted(glob.glob(os.path.join(folder, video_dir, 'vis', '*.jpg')))
    frames_raw = []
    frames_crop = []
    for framefile, bbox in zip(framefiles, bboxes):
      frame_raw = cv2.imread(framefile)
      frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
      frames_raw.append(frame_raw)
      # Crop frame based on bounding box location
      w = int(bbox[3]*2)
      h = int(bbox[4]*2)
      x = max(int(bbox[1]-bbox[3]), 0)
      y = max(int(bbox[2]-bbox[4]), 0)
      frame_rgb = frame_raw[y:y+h, x:x+w]
      if frame_rgb.size != 0:
        frame_rgb = cv2.resize(frame_rgb, (width, height))
        frames_crop.append(frame_rgb)
    frames_crop = np.asarray(frames_crop)
    frames_raw = np.asarray(frames_raw)
    videos.append(frames_crop)
    videos_raw.append(frames_raw)
    video_seq_lens.append(len(frames_crop))
    skeletons.append(skeleton)
    print('Video {} Total {} frame'.format(video_dir, len(frames_crop))) 
  max_seq_len = max(video_seq_lens)
  videos = np.asarray([pad_zeros(x, max_seq_len) for x in videos])
  #videos_raw = np.asarray([pad_zeros(x, max_seq_len) for x in videos_raw])
  return videos, video_seq_lens, videos_raw, skeletons

def create_dataset(videos, seq_lens, batch_size, num_steps,
                   num_context_steps, context_stride): 
  with tf.device('/CPU:0'):
    ds = tf.data.Dataset.from_tensor_slices((videos, seq_lens))
    ds = ds.repeat()
    ds = ds.shuffle(len(videos))
    print('[CLEA] min(seq_lens) = ', min(seq_lens))

    def sample_and_preprocess(video, seq_len):
      steps = tf.sort(tf.random.shuffle(tf.range(seq_len))[:num_steps])
      
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
      frames = (frames/127.5) - 1.0
      frames = tf.image.resize(frames, (168, 168))
      return {'frames': frames,
              'seq_lens': seq_len,
              'steps': steps}

    ds = ds.map(sample_and_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
  
  return ds