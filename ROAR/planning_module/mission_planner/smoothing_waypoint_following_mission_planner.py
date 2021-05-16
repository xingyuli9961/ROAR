from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from collections import deque
from ROAR.agent_module.agent import Agent


class SmoothingWaypointFollowingMissionPlanner(WaypointFollowingMissionPlanner):
    def __init__(self, agent: Agent, closeness_threshold=5.0, farness_threshold=10.0):
        self.closeness_threshold = closeness_threshold
        self.farness_threshold = farness_threshold
        super().__init__(agent)

    def produce_mission_plan(self) -> deque:
        mission_plan = super(SmoothingWaypointFollowingMissionPlanner, self).produce_mission_plan()
        return self._smoothen(mission_plan)

    def produce_single_lap_mission_plan(self) -> deque:
        """
        Produce a deque with waypoints that are greater than closeness_threshold apart
        but less than farness threshold apart
        Returns:

        """
        mission_plan: deque = super(SmoothingWaypointFollowingMissionPlanner, self).produce_single_lap_mission_plan()
        return self._smoothen(mission_plan)

    def _smoothen(self, deq):
        final_plan = deque(maxlen=len(deq))
        final_plan.append(deq.popleft())
        # ignore waypoints that are too close to the previous one
        while len(deq) > 0:
            transform = deq.popleft()
            latest_loc = final_plan[-1].location
            if transform.location.distance(latest_loc) > self.closeness_threshold:
                final_plan.append(transform)

        # TODO generate waypoints that are too far apart

        return final_plan
