import os
import sys
import csv
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
from scipy.spatial.transform import Rotation as R

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
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)
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
        return gg

    def collision_detection(self, gg, cloud):
        self.get_logger().info("Performing collision detection...")
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        self.get_logger().info("Collision detection complete.")
        return gg

    def vis_grasps(self, gg, cloud):
        self.get_logger().info("Visualizing and saving grasps...")

        # Sort grasps by score and get the top 5 highest score grasps
        gg.nms()
        gg.sort_by_score()
        top_5_grasps = gg[:5]  # Get top 5 highest scoring grasps

        # Define colors for the top 5 grasps
        grasp_colors = [
            [1.0, 0.0, 0.0],  # Top 1 - Red
            [0.0, 0.0, 1.0],  # Top 2 - Blue
            [1.0, 0.5, 0.0],  # Top 3 - Orange
            [0.0, 1.0, 0.0],  # Top 4 - Green
            [1.0, 1.0, 0.0]   # Top 5 - Yellow
        ]

        # Apply colors to top 5 grasp poses
        grippers_highest_scores = []
        for i, grasp in enumerate(top_5_grasps):
            grasp_geom = grasp.to_open3d_geometry()
            grasp_geom.paint_uniform_color(grasp_colors[i])  # Apply color
            grippers_highest_scores.append(grasp_geom)

        geometries_highest_scores = [cloud, *grippers_highest_scores]

        vis_highest_scores = o3d.visualization.Visualizer()
        vis_highest_scores.create_window(visible=False)
        for geom in geometries_highest_scores:
            vis_highest_scores.add_geometry(geom)

        # Set view control for top 5 grasp poses
        view_control = vis_highest_scores.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.8)

        vis_highest_scores.poll_events()
        vis_highest_scores.update_renderer()

        # Save the image of the top 5 highest score grasp poses
        save_path_highest_scores = '/home/moniesha/graspnet-baseline/doc/highest_score_grasp.png'
        vis_highest_scores.capture_screen_image(save_path_highest_scores)
        vis_highest_scores.destroy_window()

        self.get_logger().info(f"Top 5 highest score grasp visualization saved at: {save_path_highest_scores}")

        # Visualizing the top 50 grasps (unchanged)
        gg = gg[:50]  # Top 50 grasps
        grippers = gg.to_open3d_geometry_list()
        #o3d.visualization.draw_geometries([cloud, *grippers])

        output_file = '/home/moniesha/graspnet-baseline/doc/top_50_grasp_poses.csv'
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            self.get_logger().info(f"Saving top 50 grasp poses to: {output_file}")
            writer.writerow(['Grasp ID', 'Translation', 'Quaternion'])
            for i, grasp in enumerate(gg):
                translation = grasp.translation
                rotation_matrix = grasp.rotation_matrix
                
                # Rotation transformations
                B = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                C = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                rotation_matrix1 = np.dot(rotation_matrix, B)
                rotation_matrix2 = np.dot(rotation_matrix1, C)
                rotation = R.from_matrix(rotation_matrix2).as_quat()
                
                writer.writerow([i + 1, ','.join(map(str, translation)), ','.join(map(str, rotation))])

        # Delete unnecessary files after saving CSV
        data_dir = '/home/moniesha/graspnet-baseline/doc/example_data'
        files_to_delete = ['color.png', 'depth.png', 'meta.mat']
        for filename in files_to_delete:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                self.get_logger().info(f"Deleted file: {file_path}")
            else:
                self.get_logger().warning(f"File not found, skipping: {file_path}")

        # Visualize and save the top 50 grasps
        geometries = [cloud, *grippers]
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geom in geometries:
            vis.add_geometry(geom)

        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.8)

        vis.poll_events()
        vis.update_renderer()
        save_path = '/home/moniesha/graspnet-baseline/doc/top_50_grasps.png'
        vis.capture_screen_image(save_path)
        vis.destroy_window()

        self.get_logger().info(f"Grasp visualization saved at: {save_path}")

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
    node = GraspPublisherNode()
    data_dir = '/home/moniesha/graspnet-baseline/doc/example_data'
    node.demo(data_dir)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
