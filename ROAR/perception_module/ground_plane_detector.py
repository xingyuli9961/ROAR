from ROAR.agent_module.agent import Agent
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from typing import Optional, Any
import open3d as o3d
import time, cv2


class GroundPlaneDetector(DepthToPointCloudDetector):
    def __init__(self, agent: Agent, knn: int = 200, vis: bool=False, **kwargs):
        super().__init__(agent, **kwargs)
        self.reference_norm: Optional[np.ndarray] = np.array(
            [-0.00000283, -0.00012446, 0.99999999])
        self.knn = knn
        self.step = 1
        self.horizon_index = 200
        indices = self.compute_vectors_near_me()
        self.f1, self.f2, self.f3, self.f4 = indices
        self.pcd = o3d.geometry.PointCloud()
        self.visualize = vis
        self.pred_roll = 0
        self.road_normal = np.array([0, 1, 0])

    def run_in_series(self) -> Any:
        t1 = time.time()
        points = super(GroundPlaneDetector, self).run_in_series()  # Nx3

        # Numpy Variation
        x = points[self.f3, :] - points[self.f4, :]
        y = points[self.f1, :] - points[self.f2, :]
        computed_normals = self.normalize_v3(np.cross(x, y))
        d1, d2 = self.idx.shape
        normals = np.zeros((d1 * d2, 3))
        normals[self.rand_inds, :] = computed_normals
        seed_point = (int(d2/2), d1 - 10)
        curr_img = normals.reshape(d1, d2, 3).astype(np.float32)
        t2 = time.time()

        # Open3D KNN
        self.pcd.points = o3d.utility.Vector3dVector(computed_normals)
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        k, idx, _ = pcd_tree.search_knn_vector_3d(
            self.pcd.points[self.seed_index], 200)

        self.road_normal = np.mean(computed_normals[idx, :], axis=0)

        if True:
            # OpenCV FloodFill
            # _, retval, _, _ = cv2.floodFill(image=curr_img,
            #                                 seedPoint=seed_point,
            #                                 newVal=(0, 0, 0),
            #                                 loDiff=(0.01, 0.01, 0.01),
            #                                 upDiff=(0.01, 0.01, 0.01),
            #                                 mask=None)
            # bool_matrix = np.mean(retval, axis=2) == 0

            bool_matrix = np.zeros(d1 * d2)
            bool_vec = np.zeros(self.rand_inds.shape[0])
            bool_vec[idx[1:]] = 1
            bool_matrix[self.rand_inds] = bool_vec
            bool_matrix = bool_matrix.reshape(d1, d2)
            t3 = time.time()

            # Numpy Variation
            d1 = self.agent.front_depth_camera.image_size_y
            d2 = self.agent.front_depth_camera.image_size_x
            bool_zeros = np.zeros((d1, d2))
            bool_zeros[self.idx, self.jdx] = bool_matrix * 1
            bool_matrix = bool_zeros

            color_image = self.agent.front_rgb_camera.data.copy()
            color_image[bool_matrix > 0] = 255
            # print("FPS1: ", 1/(t2 - t1), ", FPS2: ", 1/(t3 - t2))

            road_normal = np.array(self.road_normal)
            road_dot = np.dot(road_normal, np.array([0, 1, 0]))
            true_roll = self.agent.vehicle.transform.rotation.roll
            # pred_roll = (np.arccos(road_dot) - np.pi / 2) * 100
            pred_roll = np.arccos(road_dot) * 180 / np.pi - 90
            self.pred_roll = pred_roll

            if self.visualize:
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                fontScale              = 0.7
                fontColor              = (255,255,255)
                lineType               = 2
                cv2.putText(color_image, f'{pred_roll}', (10, 30),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.putText(color_image, f'{true_roll}', (10, 65),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.putText(color_image, f'{road_normal}', (10, 100),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                cv2.imshow('Color', color_image)
                cv2.waitKey(1)
        else:
            color_image = curr_img + curr_img.min()
            color_image = 255 * curr_img / curr_img.max()
            color_image = color_image.astype(np.uint8)
            radius = 10
            color = (255, 255, 255)
            thickness = 2
            color_image = cv2.circle(color_image, seed_point, radius, color, thickness)

            if self.visualize:
                cv2.imshow('Color', color_image)
                cv2.waitKey(1)

    @staticmethod
    def construct_pointcloud(points) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        return pcd

    def compute_reference_norm(self, pcd: o3d.geometry.PointCloud):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
        [k, idx, _] = pcd_tree.search_knn_vector_3d(self.agent.vehicle.transform.location.to_array(),
                                                    knn=self.knn)  # find points around me
        points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3
        u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)  # use svd to find normals of points
        self.reference_norm = vh[2, :]

    @staticmethod
    def normalize_v3(arr):
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        lens[lens <= 0] = 1
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def compute_vectors_near_me(self):
        d1 = self.agent.front_depth_camera.image_size_y
        d2 = self.agent.front_depth_camera.image_size_x
        idx, jdx = np.indices((d1, d2))

        idx_back = np.clip(idx - 1, 0, idx.max())[self.horizon_index:, :].flatten()
        idx_front = np.clip(idx + 1, 0, idx.max())[self.horizon_index:, :].flatten()
        jdx_back = np.clip(jdx - 1, 0, jdx.max())[self.horizon_index:, :].flatten()
        jdx_front = np.clip(jdx + 1, 0, jdx.max())[self.horizon_index:, :].flatten()
        idx_flat = idx[self.horizon_index:, :].flatten()
        jdx_flat = jdx[self.horizon_index:, :].flatten()

        # rand_idx = np.random.choice(np.arange(idx.shape[0]), size=d1*d2, replace=False)
        f1 = (idx_front * d2 + jdx_flat)[::self.step]  # [rand_idx]
        f2 = (idx_back * d2 + jdx_flat)[::self.step]  # [rand_idx]
        f3 = (idx_flat * d2 + jdx_front)[::self.step]  # [rand_idx]
        f4 = (idx_flat * d2 + jdx_back)[::self.step]  # [rand_idx]

        # indices = np.arange(len(f1))
        indices = np.random.choice(np.arange(len(f1)), 5000)
        f1, f2, f3, f4 = f1[indices], f2[indices], f3[indices], f4[indices]
        self.rand_inds = indices
        self.idx = idx[self.horizon_index:, :]
        self.jdx = jdx[self.horizon_index:, :]

        d1, d2 = self.idx.shape
        seed_point = (d1 - 10, int(d2/2))
        true_seed = [self.idx[seed_point], self.jdx[seed_point]]
        seed = [idx_flat[indices][0], jdx_flat[indices][0]]
        self.seed_index = 0
        seed_index = 0

        for i, j in zip(idx_flat[indices], jdx_flat[indices]):
            orig_diff_vec = np.array(seed) - np.array(true_seed)
            orig_diff_dist = np.linalg.norm(orig_diff_vec, 2)
            new_diff_vec = np.array([i, j]) - np.array(true_seed)
            new_diff_dist = np.linalg.norm(new_diff_vec, 2)
            if new_diff_dist < orig_diff_dist:
                seed[0], seed[1] = i, j
                self.seed_index = seed_index
            seed_index += 1

        print()
        print("[INIT] True Seed:", true_seed, ", Approx. Seed:", seed)
        print()

        return f1, f2, f3, f4
