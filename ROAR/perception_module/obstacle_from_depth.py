from typing import Any

from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
import open3d
from pydantic import BaseModel, Field
import time
from PIL import Image
from skimage.measure import block_reduce


class ObstacleFromDepth(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)
        config = ObstacleFromDepthConfig.parse_file(self.agent.agent_settings.obstacle_from_depth_config_path)
        self.max_detectable_distance = kwargs.get("max_detectable_distance", config.max_detectable_distance)
        self.max_incline_normal = kwargs.get("max_incline_normal", config.max_incline_normal)
        self.min_obstacle_height = kwargs.get("max_obstacle_height", config.min_obstacle_height)
        self.voxel_downsample_rate = kwargs.get("voxel_downsample_rate", config.voxel_downsample_rate)
        self.depth_scale = kwargs.get("depth_scale", config.depth_scale)

    def run_in_series(self, **kwargs) -> Any:
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data.copy()
            indices = np.indices(depth_img.shape)
            cols = indices[0].flatten()
            rows = indices[1].flatten()
            depth_img_1d = depth_img[cols, rows]
            wat1 = depth_img_1d * rows * self.depth_scale
            wat2 = depth_img_1d * cols * self.depth_scale
            wat3 = depth_img.flatten() * self.depth_scale
            raw_p2d = np.vstack([wat1, wat2, wat3])

            # only take close indicies
            close_indices = np.where(raw_p2d[2, :] < self.max_detectable_distance)
            raw_p2d = np.squeeze(raw_p2d[:, close_indices])
            cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d
            cords_xyz_1 = np.vstack([
                cords_y_minus_z_x[0, :],
                -cords_y_minus_z_x[1, :],
                -cords_y_minus_z_x[2, :],
                np.ones((1, np.shape(cords_y_minus_z_x)[1]))
            ])

            points = self.agent.vehicle.transform.get_matrix() @ cords_xyz_1
            points = points.T[:, :3]

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_downsample_rate)
            pcd.estimate_normals()
            pcd.normalize_normals()

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            abs_normals = np.abs(normals)
            obstacles_mask = abs_normals[:, 1] < self.max_incline_normal
            obstacle_below_height_mask = np.abs(
                points[:, 1]) < self.agent.vehicle.transform.location.y + self.min_obstacle_height
            mask = obstacles_mask & obstacle_below_height_mask

            self.agent.kwargs["point_cloud_obstacle_from_depth"] = points
            self.agent.kwargs["obstacle_coords"] = points[mask]
            self.agent.kwargs["ground_coords"] = points[~mask]
            return self.agent.kwargs["obstacle_coords"]


class ObstacleFromDepthConfig(BaseModel):
    depth_scale: float = Field(default=1000, description="scaling depth [0 - 1] to real world unit")
    max_detectable_distance: float = Field(default=300, description="Max detectable distance in meter")
    max_incline_normal: float = Field(default=0.5)
    min_obstacle_height: float = Field(default=3)
    update_interval: float = Field(default=0.1)
    voxel_downsample_rate: float = Field(default=0.2)
