from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

CONFIG.CA_path = "/home/lin10/projects/SkatingApp/certs.pem"

# Path to downloaded videos
CONFIG.video_buffer_dir = "/home/lin10/projects/SkatingApp/cache/video"

# Path to scripts
CONFIG.alphapose_script = "/home/lin10/projects/SkatingApp/engine/alphapose/run_alphapose.sh"
CONFIG.alignment_script = "/home/lin10/projects/SkatingApp/engine/VideoAlignment/run_alignment.sh"
CONFIG.jump_script = "/home/lin10/projects/SkatingApp/engine/JumpDetection/run_jump_detection.sh"

# Path to align data
CONFIG.align_data_dir = "/home/lin10/projects/SkatingApp/cache/align/data/flip"
# Path to jump data
CONFIG.jump_alphapose_result_dir = "/home/lin10/projects/SkatingApp/cache/jump/data/alphapose"
CONFIG.jump_pkl_result_dir = "/home/lin10/projects/SkatingApp/cache/jump/data/pkl"

# Path to results
CONFIG.result_dir = "/home/lin10/projects/SkatingApp/results"