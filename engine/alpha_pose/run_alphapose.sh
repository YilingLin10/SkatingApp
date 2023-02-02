#!/bin/bash
cd /home/lin10/projects/SkatingApp/engine/alpha_pose
CUDA_LAUNCH_BLOCKING=1 python ./videopose.py --video $2

if [[ $1 == "jump" ]]; then  
    mv /home/lin10/projects/SkatingApp/engine/alpha_pose/results/alpha_pose_$2 /home/lin10/projects/SkatingApp/cache/jump/data/alphapose
elif [[ $1 == "align" ]]; then
    mv /home/lin10/projects/SkatingApp/engine/alpha_pose/results/alpha_pose_$2 /home/lin10/projects/SkatingApp/cache/align/data/flip/$3
else
    echo $1
    echo "Command not found."
fi