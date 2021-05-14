from typing import Tuple
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.utilities_module.data_structures_models import SensorsData, Transform
from pathlib import Path
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.control_module.pid_controller import PIDController
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
import numpy as np
import cv2


class LineDetectAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        self.plan_lst = list(self.mission_planner.produce_single_lap_mission_plan())
        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self, threaded=True)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        self.add_threaded_module(self.occupancy_map)
        self.kwargs = kwargs
        self.interval = self.kwargs.get('interval', 20)
        self.look_back = self.kwargs.get('look_back', 5)
        self.look_back_max = self.kwargs.get('look_back_max', 10)
        self.thres = self.kwargs.get('thres', 1e-3)
        self.int_counter = 0
        self.counter = 0
        self.finished = False
        self._get_next_bbox()

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(LineDetectAgent, self).run_step(sensors_data, vehicle)
        self.counter += 1
        if not self.finished:
            crossed, dist = self.bbox.has_crossed(vehicle.transform)
            if crossed:
                self.int_counter += 1
                self._get_next_bbox()

        option = "obstacle_coords"  # ground_coords, obstacle_coords, point_cloud_obstacle_from_depth
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update_async(points)

        strip_locs = self.occupancy_map.cord_translation_from_world(
            self.bbox.get_visualize_locs(size=20) * self.occupancy_map.world_coord_resolution)

        map_copy = self.occupancy_map.get_map(transform=self.vehicle.transform, view_size=(400, 400),
                                              vehicle_value=1,
                                              arbitrary_locations=strip_locs, arbitrary_point_value=1)

        cv2.imshow("map_copy", cv2.resize(map_copy, dsize=(500, 500)))
        cv2.waitKey(1)
        return VehicleControl()

    def _get_next_bbox(self):
        # make sure no index out of bound error
        curr_lb = self.look_back
        curr_idx = self.int_counter * self.interval
        while curr_idx + curr_lb < len(self.plan_lst):
            if curr_lb > self.look_back_max:
                self.int_counter += 1
                curr_lb = self.look_back
                curr_idx = self.int_counter * self.interval
                continue

            t1 = self.plan_lst[curr_idx]
            t2 = self.plan_lst[curr_idx + curr_lb]

            dx = t2.location.x - t1.location.x
            dz = t2.location.z - t1.location.z
            if abs(dx) < self.thres and abs(dz) < self.thres:
                curr_lb += 1
            else:
                self.bbox = LineBBox(t1, t2)
                return
                # no next bbox
        print("finished all the iterations!")
        self.finished = True


class LineBBox(object):
    def __init__(self, transform1: Transform, transform2: Transform) -> None:
        self.x1, self.z1 = transform1.location.x, transform1.location.z
        self.x2, self.z2 = transform2.location.x, transform2.location.z
        self.pos_true = True
        self.thres = 1e-2
        self.eq = self._construct_eq()
        self.strip_list = None

        if self.eq(self.x1, self.z1) > 0:
            self.pos_true = False

    def _construct_eq(self):
        dz, dx = self.z2 - self.z1, self.x2 - self.x1

        if abs(dz) < self.thres:
            # print("vertical strip")
            def vertical_eq(x, z):
                return x - self.x2

            return vertical_eq
        elif abs(dx) < self.thres:
            # print("horizontal strip")
            def horizontal_eq(x, z):
                return z - self.z2

            return horizontal_eq

        slope_ = dz / dx
        self.slope = -1 / slope_
        # print("tilted strip with slope {}".format(self.slope))
        self.intercept = -(self.slope * self.x2) + self.z2

        def linear_eq(x, z):
            return z - self.slope * x - self.intercept

        return linear_eq

    def has_crossed(self, transform: Transform):
        x, z = transform.location.x, transform.location.z
        dist = self.eq(x, z)
        return dist > 0 if self.pos_true else dist < 0, dist

    def get_visualize_locs(self, size=10):
        # if self.strip_list is not None:
        #     return self.strip_list

        name = self.eq.__name__
        if name == 'vertical_eq':
            xs = np.repeat(self.x2, size)
            zs = np.arange(self.z2 - (size // 2), self.z2 + (size // 2))
        elif name == 'horizontal_eq':
            xs = np.arange(self.x2 - (size // 2), self.x2 + (size // 2))
            zs = np.repeat(self.z2, size)
        else:
            range_ = size * np.cos(np.arctan(self.slope))
            xs = np.linspace(self.x2 - range_ / 2, self.x2 + range_ / 2, num=size)
            zs = self.slope * xs + self.intercept
            # print(np.vstack((xs, zs)).T)

        self.strip_list = np.vstack((xs, zs)).T
        return self.strip_list
