from ROAR.agent_module.agent import Agent
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from typing import Optional, Any
import open3d as o3d
import time, cv2


class ColorPlaneDetector(DepthToPointCloudDetector):
    def __init__(self, agent: Agent, knn: int = 200, **kwargs):
        super().__init__(agent, **kwargs)
        self.reference_norm: Optional[np.ndarray] = np.array(
            [-0.00000283, -0.00012446, 0.99999999])
        self.knn = knn
        self.step = 1
        self.horizon_index = 330

    def run_in_series(self) -> Any:
        t1 = time.time()
        d1 = self.agent.front_depth_camera.image_size_y
        d2 = self.agent.front_depth_camera.image_size_x
        color_image = self.agent.front_rgb_camera.data.copy()
        seed_point = (int(d2/2), d1 - 10)

        # OpenCV FloodFill
        _, retval, _, _ = cv2.floodFill(image=color_image,
                                        seedPoint=seed_point,
                                        newVal=(0, 0, 0),
                                        loDiff=(0.1, 0.1, 0.1),
                                        upDiff=(0.1, 0.1, 0.1),
                                        mask=None)

        t2 = time.time()

        # Filling in Points
        bool_matrix = np.mean(retval, axis=2) == 0
        color_image = self.agent.front_rgb_camera.data.copy()
        color_image[bool_matrix > 0] = 255

        radius = 10
        color = (255, 255, 255)
        thickness = 2
        color_image = cv2.circle(color_image, seed_point, radius, color, thickness)
        print("[COLOR PLANE] FPS1: ", 1/(t2 - t1))

        cv2.imshow('Color', color_image)
        cv2.waitKey(1)


    def compute_reference_norm(self, pcd: o3d.geometry.PointCloud):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
        [k, idx, _] = pcd_tree.search_knn_vector_3d(self.agent.vehicle.transform.location.to_array(),
                                                    knn=self.knn)  # find points around me
        points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3
        u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)  # use svd to find normals of points
        self.reference_norm = vh[2, :]
