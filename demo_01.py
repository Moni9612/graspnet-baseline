import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import torch
from graspnetAPI import GraspGroup
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    try:
        logger.info("Initializing the model...")
        net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        logger.info("Loading checkpoint from %s...", cfgs.checkpoint_path)
        checkpoint = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info("-> loaded checkpoint %s (epoch: %d)", cfgs.checkpoint_path, start_epoch)
        net.eval()
        return net
    except Exception as e:
        logger.error("Error in get_net: %s", e)
        raise

def get_and_process_data(data_dir):
    try:
        logger.info("Loading data from %s...", data_dir)
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        logger.info("Data processing completed successfully.")
        return end_points, cloud
    except Exception as e:
        logger.error("Error in get_and_process_data: %s", e)
        raise

def get_grasps(net, end_points):
    try:
        logger.info("Getting grasps from the network...")
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        logger.info("Grasp predictions obtained.")
        return gg
    except Exception as e:
        logger.error("Error in get_grasps: %s", e)
        raise

def collision_detection(gg, cloud):
    try:
        logger.info("Starting collision detection...")
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        logger.info("Collision detection completed.")
        return gg
    except Exception as e:
        logger.error("Error in collision_detection: %s", e)
        raise

def vis_grasps(gg, cloud):
    try:
        logger.info("Visualizing grasps...")
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        logger.info("Grasp visualization completed.")
    except Exception as e:
        logger.error("Error in vis_grasps: %s", e)
        raise

def demo(data_dir):
    try:
        logger.info("Starting demo...")
        net = get_net()
        end_points, cloud = get_and_process_data(data_dir)
        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        vis_grasps(gg, cloud)
        logger.info("Demo completed successfully.")
    except Exception as e:
        logger.error("Error in demo: %s", e)

if __name__ == '__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)

