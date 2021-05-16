from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.rl_local_planner import RLLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
import numpy as np
from typing import Any
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.smoothing_waypoint_following_mission_planner import \
    SmoothingWaypointFollowingMissionPlanner
from ROAR.utilities_module.line_bbox import LineBBox
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
from typing import Optional


class RLLocalPlannerAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mission_planner = SmoothingWaypointFollowingMissionPlanner(agent=self)
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.local_planner = LoopSimpleWaypointFollowingLocalPlanner(agent=self,
                                                                     behavior_planner=self.behavior_planner,
                                                                     mission_planner=self.mission_planner,
                                                                     controller=self.controller)
        self.occupancy_map = OccupancyGridMap(agent=self)
        occu_map_file_path = Path("../ROAR_Sim/data/easy_map_cleaned_global_occu_map.npy")
        self.occupancy_map.load_from_file(occu_map_file_path)
        self.bbox: Optional[LineBBox] = None
        self._get_next_bbox()

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(RLLocalPlannerAgent, self).run_step(vehicle=vehicle,
                                                  sensors_data=sensors_data)
        self.local_planner.run_in_series()
        self._get_next_bbox()

        if self.kwargs.get("control") is not None:
            return self.kwargs.get("control")
        return VehicleControl()

    def _get_next_bbox(self):
        index = self.local_planner.get_curr_waypoint_index()
        t1 = self.local_planner.way_points_queue[index]
        t2 = self.local_planner.way_points_queue[(index + 1) % len(self.local_planner.way_points_queue)]
        self.bbox = LineBBox(t1, t2)

    def get_obs(self) -> np.ndarray:
        strip_locs = self.occupancy_map.cord_translation_from_world(
            self.bbox.get_visualize_locs(size=20) * self.occupancy_map.world_coord_resolution)
        full_occu_map = self.occupancy_map.get_map(transform=self.vehicle.transform,
                                                   view_size=(100, 100),
                                                   arbitrary_locations=strip_locs,
                                                   arbitrary_point_value=-10)
        occu_strip_locs = np.where(full_occu_map == -10)
        obs = np.zeros(shape=(100, 100, 3), dtype=np.uint8)
        obs[:, :, 0] = (full_occu_map.clip(0, 1) * 255).astype(np.uint8)
        obs[occu_strip_locs[0], occu_strip_locs[1], 1] = 1
        obs[:, :, 1] = (obs[:, :, 1] * 255).astype(np.uint8)
        return obs
