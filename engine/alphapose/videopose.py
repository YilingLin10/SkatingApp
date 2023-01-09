import os
import time

from common.arguments import parse_args
from common.camera import *
from common.utils import Timer, add_path
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    detector_2d = get_detector_2d(args.detector_2d)

    assert detector_2d, 'detector_2d should be in ({alpha, hr, open}_pose)'

    # 2D kpts loads or generate
    if not args.input_npz:
        video_name = args.viz_video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts']  # (N, 17, 2)


def inference_video(detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()
    video_path = f'/home/lin10/projects/SkatingApp/cache/video/{args.input_video}.mp4'
    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    video_name = basename[:basename.rfind('.')]
    args.viz_video = video_path
    args.viz_output = f'{dir_name}/{args.detector_2d}_{video_name}.mp4'

    with Timer(video_path):
        main(args)

if __name__ == '__main__':
    inference_video('alpha_pose')
    torch.cuda.empty_cache()