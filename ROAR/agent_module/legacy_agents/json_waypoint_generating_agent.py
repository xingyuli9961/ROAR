from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path
from ROAR.perception_module.legacy.ground_plane_detector import GroundPlaneDetector


class JSONWaypointGeneratigAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.output_file_path: Path = self.output_folder_path / "easy_map_waypoint_t.txt"
        self.output_file = self.output_file_path.open('w')
        # self.depth_to_pointcloud_detector = DepthToPointCloudDetector(self)
        self.gpd = GroundPlaneDetector(self, should_compute_global_pointcloud=False)

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(JSONWaypointGeneratigAgent, self).run_step(sensors_data=sensors_data,
                                                         vehicle=vehicle)
        try:
            self.gpd.run_in_series()
            ground_points = self.kwargs["ground_points"]
            # using the next waypoint to find the line

        except Exception as e:
            self.logger.error(e)

        return VehicleControl()
