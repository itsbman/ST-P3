from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour


def visualize_scene():
    plt.subplot()
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(make_contour(showing))
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()


checkpoint_path = '/home/user/data/abi/ST-P3/checkpoints/STP3_plan.ckpt'
dataroot = '/home/user/data/Dataset/nuscenes'

trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
print(f'Loaded weights from \n {checkpoint_path}')
trainer.eval()

device = torch.device('cpu')
trainer.to(device)
model = trainer.model

cfg = model.cfg
cfg.GPUS = "[0]"
cfg.BATCHSIZE = 1
cfg.LIFT.GT_DEPTH = False
cfg.DATASET.DATAROOT = dataroot
cfg.DATASET.MAP_FOLDER = dataroot
print(cfg.DATASET.VERSION)

dataroot = cfg.DATASET.DATAROOT
nworkers = cfg.N_WORKERS
nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
# valdata = FuturePredictionDataset(nusc, 1, cfg)
# valloader = torch.utils.data.DataLoader(
#     valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
# )

# n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
# hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
# metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
# future_second = int(cfg.N_FUTURE_FRAMES / 2)

# if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
#     metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

# if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
#     metric_hdmap_val = []
#     for i in range(len(hdmap_class)):
#         metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

# if cfg.INSTANCE_SEG.ENABLED:
#     metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

# if cfg.PLANNING.ENABLED:
#     metric_planning_val = []
#     for i in range(future_second):
#         metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))

print("processing dataset")
valdata = FuturePredictionDataset(nusc, 1, cfg)

# scene_token = b51869782c0e464b8021eb798609f35f
# valdata.ixes[4375]
# valdata.indices[3503:3534]
data_sample = valdata[3503]
print(len(valdata))
