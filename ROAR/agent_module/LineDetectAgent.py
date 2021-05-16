from typing import Tuple
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.utilities_module.data_structures_models import SensorsData, Transform
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from pathlib import Path
from ROAR.planning_module.mission_planner.smoothing_waypoint_following_mission_planner import SmoothingWaypointFollowingMissionPlanner
from ROAR.control_module.pid_controller import PIDController
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
import numpy as np
import cv2
from ROAR.utilities_module.line_bbox import LineBBox


class LineDetectAgent(Agent):
    """
    Note that this agent only support one lap
    """

    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.mission_planner = SmoothingWaypointFollowingMissionPlanner(agent=self)
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.local_planner = LoopSimpleWaypointFollowingLocalPlanner(agent=self,
                                                                     behavior_planner=self.behavior_planner,
                                                                     mission_planner=self.mission_planner,
                                                                     controller=self.controller)
        self.plan_lst = list(self.mission_planner.produce_mission_plan())
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
        return self.local_planner.run_in_series()

    def _get_next_bbox(self):
        index = self.local_planner.get_curr_waypoint_index()
        t1 = self.local_planner.way_points_queue[index]
        t2 = self.local_planner.way_points_queue[(index + 1) % len(self.local_planner.way_points_queue)]
        self.bbox = LineBBox(t1, t2)
