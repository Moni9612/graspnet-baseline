import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import logging
import rclpy
from rclpy.node import Node

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# Set up Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROS 2 node setup
class GraspPublisherNode(Node):
    def __init__(self):
        super().__init__('grasp_publisher_node')

        # Initialize logger
        self.get_logger().info('GraspPublisherNode has been started.')

        # Argument parser setup
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
        parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
        parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
        parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
        parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
        self.cfgs = parser.parse_args()

    def get_net(self):
        self.get_logger().info("Initializing model...")
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        self.get_logger().info("Loading checkpoint...")
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        self.get_logger().info("-> Loaded checkpoint %s (epoch: %d)" % (self.cfgs.checkpoint_path, start_epoch))
        net.eval()
        self.get_logger().info("Model is set to evaluation mode.")
        return net

    def get_and_process_data(self, data_dir):
    	self.get_logger().info(f"Loading data from directory: {data_dir}")
    	color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    	depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    	workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    	#meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    	meta = scio.loadmat('/home/moniesha/graspnet-baseline/doc/example_data/meta.mat')
    	intrinsic = meta['intrinsic_matrix']
    	factor_depth = meta['factor_depth']

    	camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    	cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    	mask = (workspace_mask & (depth > 0))
    	cloud_masked = cloud[mask]
    	color_masked = color[mask]

    	if len(cloud_masked) >= self.cfgs.num_point:
        	idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
    	else:
        	idxs1 = np.arange(len(cloud_masked))
        	idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
        	idxs = np.concatenate([idxs1, idxs2], axis=0)
    	cloud_sampled = cloud_masked[idxs]
    	color_sampled = color_masked[idxs]

    	cloud = o3d.geometry.PointCloud()
    	cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    	cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    	end_points = dict()
    	cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    	cloud_sampled = cloud_sampled.to(device)
    	end_points['point_clouds'] = cloud_sampled
    	end_points['cloud_colors'] = color_sampled

    	self.get_logger().info(f"Data processing successful. Number of points: {len(cloud_masked)}")
    	return end_points, cloud


    def get_grasps(self, net, end_points):
        self.get_logger().info("Generating grasp predictions...")
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        self.get_logger().info("Grasp predictions generated successfully.")
        return gg

    def collision_detection(self, gg, cloud):
        self.get_logger().info("Performing collision detection...")
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        self.get_logger().info("Collision detection complete.")
        return gg

    def vis_grasps(self, gg, cloud):
        self.get_logger().info("Visualizing grasps...")
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        self.get_logger().info("Grasps visualization complete.")

    def demo(self, data_dir):
        self.get_logger().info("Starting demo...")
        net = self.get_net()
        end_points, cloud = self.get_and_process_data(data_dir)
        gg = self.get_grasps(net, end_points)
        if self.cfgs.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        self.vis_grasps(gg, cloud)
        self.get_logger().info("Demo completed successfully.")

def main():
    rclpy.init()

    # Create the ROS 2 Node
    node = GraspPublisherNode()

    # Run the demo
    data_dir = '/home/moniesha/graspnet-baseline/doc/example_data'
    node.demo(data_dir)

    rclpy.shutdown()

if __name__ == '__main__':
    main()

