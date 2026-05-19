"""nuPlan closed-loop planner wrapper for DOOR-RL checkpoints.

This module is deliberately safe to import without ``nuplan-devkit``. When the
devkit is installed and on ``PYTHONPATH``, ``DoorRLNuPlanPlanner`` implements
the official ``AbstractPlanner`` interface and can be passed to
``nuplan.planning.script.run_simulation.run_simulation``.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Optional, Type

import torch

from doorrl.adapters.base import TokenizationSpec
from doorrl.adapters.nuplan_adapter import NuPlanClosedLoopAdapter
from doorrl.config import DoorRLConfig
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch

try:  # pragma: no cover - exercised only when nuplan-devkit is installed.
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
    from nuplan.planning.simulation.planner.abstract_planner import (
        AbstractPlanner,
        PlannerInitialization,
        PlannerInput,
    )
    from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

    _NUPLAN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - default in the lightweight env.
    AbstractPlanner = object  # type: ignore[misc, assignment]
    PlannerInitialization = Any  # type: ignore[misc, assignment]
    PlannerInput = Any  # type: ignore[misc, assignment]
    AbstractTrajectory = Any  # type: ignore[misc, assignment]
    DetectionsTracks = object  # type: ignore[misc, assignment]
    Observation = object  # type: ignore[misc, assignment]
    EgoState = Any  # type: ignore[misc, assignment]
    Point2D = None  # type: ignore[assignment]
    StateSE2 = None  # type: ignore[assignment]
    StateVector2D = None  # type: ignore[assignment]
    SemanticMapLayer = None  # type: ignore[assignment]
    InterpolatedTrajectory = None  # type: ignore[assignment]
    _NUPLAN_IMPORT_ERROR = exc


class NuPlanClosedLoopUnavailable(RuntimeError):
    """Raised when the optional nuPlan devkit dependency is unavailable."""


_CONDITION_TO_VARIANT = {
    "wm_object": "object_only",
    "wm_decoupled_no_vis": "object_relation_decoupled",
    "bc": "object_only",
}


class DoorRLNuPlanPlanner(AbstractPlanner):  # type: ignore[misc]
    """Oracle-token DOOR-RL planner for nuPlan simulation.

    The planner consumes nuPlan's structured ego/tracked-object observation,
    converts it to the existing DOOR-RL token schema, evaluates a Stage-1
    checkpoint, then rolls the predicted 2-D action into a short ego trajectory.
    It does not use future GT and should be configured with standard nuPlan
    closed-loop metrics.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        condition: str,
        horizon_seconds: float = 8.0,
        sampling_time: float = 0.25,
        max_speed: float = 15.0,
        speed_scale: float = 2.0,
        yaw_rate_scale: float = 0.6,
        max_accel: float = 2.5,
        max_decel: float = 4.0,
        safety_projection: bool = True,
        collision_weight: float = 80.0,
        ttc_weight: float = 25.0,
        drivable_weight: float = 60.0,
        progress_weight: float = 1.0,
        smoothness_weight: float = 12.0,
        lane_center_weight: float = 8.0,
        max_comfortable_decel: float = 2.0,
        max_comfortable_jerk: float = 2.5,
        min_progress_speed: float = 2.5,
        corridor_projection: bool = True,
        corridor_candidate_limit: int = 3,
        lead_vehicle_controller: bool = False,
        lead_time_headway: float = 2.0,
        lead_min_gap: float = 8.0,
        lead_max_speed_drop: float = 2.0,
        lead_ttc_threshold: float = 4.0,
        lead_gap_margin: float = 0.0,
        ttc_proxy: bool = False,
        ttc_proxy_clearance: float = 5.0,
        ttc_proxy_horizon: float = 4.0,
        device: str | torch.device | None = None,
    ) -> None:
        if _NUPLAN_IMPORT_ERROR is not None:
            raise NuPlanClosedLoopUnavailable(
                "nuplan-devkit is not importable. Add "
                "`/mnt/volumes/cpfs/prediction/lipeinan/code/cangku/nuplan-devkit` "
                "to PYTHONPATH and install its requirements before running "
                "closed-loop simulation."
            ) from _NUPLAN_IMPORT_ERROR

        self.config = DoorRLConfig.from_json(config_path)
        self.condition = condition
        self.variant = _CONDITION_TO_VARIANT.get(condition, condition)
        self.horizon_seconds = float(horizon_seconds)
        self.sampling_time = float(sampling_time)
        self.max_speed = float(max_speed)
        self.speed_scale = float(speed_scale)
        self.yaw_rate_scale = float(yaw_rate_scale)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.safety_projection = bool(safety_projection)
        self.collision_weight = float(collision_weight)
        self.ttc_weight = float(ttc_weight)
        self.drivable_weight = float(drivable_weight)
        self.progress_weight = float(progress_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.lane_center_weight = float(lane_center_weight)
        self.max_comfortable_decel = float(max_comfortable_decel)
        self.max_comfortable_jerk = float(max_comfortable_jerk)
        self.min_progress_speed = float(min_progress_speed)
        self.corridor_projection = bool(corridor_projection)
        self.corridor_candidate_limit = int(corridor_candidate_limit)
        self.lead_vehicle_controller = bool(lead_vehicle_controller)
        self.lead_time_headway = float(lead_time_headway)
        self.lead_min_gap = float(lead_min_gap)
        self.lead_max_speed_drop = float(lead_max_speed_drop)
        self.lead_ttc_threshold = float(lead_ttc_threshold)
        self.lead_gap_margin = float(lead_gap_margin)
        self.ttc_proxy = bool(ttc_proxy)
        self.ttc_proxy_clearance = float(ttc_proxy_clearance)
        self.ttc_proxy_horizon = float(ttc_proxy_horizon)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.initialization: PlannerInitialization | None = None
        self.adapter = NuPlanClosedLoopAdapter(
            TokenizationSpec(
                raw_dim=self.config.model.raw_dim,
                max_tokens=self.config.model.max_tokens,
                max_dynamic_objects=self.config.data.max_dynamic_objects,
                max_map_tokens=self.config.data.max_map_tokens,
                max_relation_tokens=self.config.data.max_relation_tokens,
                action_dim=self.config.model.action_dim,
            ),
            reactive=True,
        )
        self.model = DoorRLModelVariant(self.config.model, ModelVariant(self.variant))
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def name(self) -> str:
        return f"doorrl_{self.condition}"

    def initialize(self, initialization: PlannerInitialization) -> None:
        self.initialization = initialization

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore[return-value]

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        item = self.adapter.convert_planner_input(current_input, self.initialization)
        batch = SceneBatch.collate([item]).to(self.device)
        output = self.model(batch)
        action = output.policy.action_mean[0].detach().float().cpu()
        ego_state = current_input.history.ego_states[-1]
        observation = self.adapter._latest_from_sequence(  # noqa: SLF001 - local duck-typed adapter helper.
            getattr(current_input.history, "observations", None)
        )
        map_api = getattr(self.initialization, "map_api", None)
        route_roadblock_ids = getattr(self.initialization, "route_roadblock_ids", None)
        return self._rollout_action_to_trajectory(
            ego_state, action, observation, map_api, route_roadblock_ids
        )

    def _rollout_action_to_trajectory(
        self,
        ego_state: EgoState,
        action: torch.Tensor,
        observation: Any | None = None,
        map_api: Any | None = None,
        route_roadblock_ids: list[str] | None = None,
    ) -> AbstractTrajectory:
        if StateSE2 is None or StateVector2D is None or InterpolatedTrajectory is None:
            raise NuPlanClosedLoopUnavailable("nuPlan trajectory classes are unavailable.")

        current_speed = max(0.0, float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
        target_speed, yaw_rate = self._action_to_control(ego_state, action)
        corridor_path = None
        if self.safety_projection:
            target_speed, yaw_rate, corridor_path = self._select_safe_control(
                ego_state=ego_state,
                nominal_target_speed=target_speed,
                nominal_yaw_rate=yaw_rate,
                current_speed=current_speed,
                observation=observation,
                map_api=map_api,
                route_roadblock_ids=route_roadblock_ids,
            )
        states, _ = self._simulate_control(
            ego_state, target_speed, yaw_rate, current_speed, corridor_path
        )
        return InterpolatedTrajectory(states)

    def _action_to_control(
        self,
        ego_state: EgoState,
        action: torch.Tensor,
    ) -> tuple[float, float]:
        current_speed = max(0.0, float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
        target_speed = max(
            0.0,
            min(self.max_speed, current_speed + float(action[0].item()) * self.speed_scale),
        )
        yaw_rate = max(
            -1.0,
            min(1.0, float(action[1].item()) * self.yaw_rate_scale),
        )
        return target_speed, yaw_rate

    def _select_safe_control(
        self,
        ego_state: EgoState,
        nominal_target_speed: float,
        nominal_yaw_rate: float,
        current_speed: float,
        observation: Any | None,
        map_api: Any | None,
        route_roadblock_ids: list[str] | None,
    ) -> tuple[float, float, list[tuple[float, float, float]] | None]:
        obstacles = self._extract_safety_obstacles(observation, ego_state)
        previous_accel = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
        previous_yaw_rate = float(getattr(ego_state.dynamic_car_state, "angular_velocity", 0.0))
        speed_candidates = self._unique_floats([
            nominal_target_speed,
            max(nominal_target_speed, current_speed, self.min_progress_speed),
            0.9 * nominal_target_speed + 0.1 * current_speed,
            min(nominal_target_speed, current_speed),
            max(min(nominal_target_speed, current_speed), self.min_progress_speed),
            0.85 * min(nominal_target_speed, current_speed),
            0.75 * min(nominal_target_speed, current_speed),
            0.65 * min(nominal_target_speed, current_speed),
            0.4 * min(nominal_target_speed, current_speed),
            0.35 * min(nominal_target_speed, current_speed),
            max(0.0, current_speed - self.max_comfortable_decel * self.horizon_seconds * 0.5),
            0.0,
        ])
        yaw_candidates = self._unique_floats([
            nominal_yaw_rate,
            0.5 * nominal_yaw_rate,
            0.0,
            max(-1.0, nominal_yaw_rate - 0.15),
            min(1.0, nominal_yaw_rate + 0.15),
            max(-1.0, nominal_yaw_rate - 0.25),
            min(1.0, nominal_yaw_rate + 0.25),
        ])
        lane_refs = self._extract_lane_center_refs(
            ego_state=ego_state,
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids,
        )
        corridor_paths = (
            self._extract_lane_corridors(ego_state, map_api, route_roadblock_ids)
            if self.corridor_projection
            else []
        )
        path_candidates: list[list[tuple[float, float, float]] | None] = (
            corridor_paths[: max(1, self.corridor_candidate_limit)] if corridor_paths else [None]
        )

        best_score = float("inf")
        best_control = (nominal_target_speed, nominal_yaw_rate, None)
        for path in path_candidates:
            path_yaw_candidates = [0.0] if path else yaw_candidates
            if path and self.lead_vehicle_controller:
                lead_speed_cap = self._corridor_lead_speed_cap(
                    ego_state=ego_state,
                    path=path,
                    obstacles=obstacles,
                    current_speed=current_speed,
                    nominal_target_speed=nominal_target_speed,
                )
                if lead_speed_cap is None:
                    path_speed_candidates = speed_candidates[:5]
                else:
                    capped = [min(speed, lead_speed_cap) for speed in speed_candidates[:5]]
                    path_speed_candidates = self._unique_floats(capped + [lead_speed_cap])[:6]
            elif path:
                path_speed_candidates = speed_candidates[:5]
            else:
                path_speed_candidates = speed_candidates
            for target_speed in path_speed_candidates:
                for yaw_rate in path_yaw_candidates:
                    _, points = self._simulate_control(
                        ego_state, target_speed, yaw_rate, current_speed, path
                    )
                    score = self._safety_score(
                        points=points,
                        obstacles=obstacles,
                        map_api=map_api,
                        nominal_target_speed=nominal_target_speed,
                        nominal_yaw_rate=nominal_yaw_rate,
                        target_speed=target_speed,
                        yaw_rate=yaw_rate,
                        current_speed=current_speed,
                        previous_accel=previous_accel,
                        previous_yaw_rate=previous_yaw_rate,
                        lane_refs=lane_refs,
                    )
                    if score < best_score:
                        best_score = score
                        best_control = (target_speed, yaw_rate, path)
        return best_control

    def _simulate_control(
        self,
        ego_state: EgoState,
        target_speed: float,
        yaw_rate: float,
        current_speed: float,
        corridor_path: list[tuple[float, float, float]] | None = None,
    ) -> tuple[list[EgoState], list[tuple[float, float, float, float]]]:
        if corridor_path:
            return self._simulate_corridor_control(
                ego_state, target_speed, yaw_rate, current_speed, corridor_path
            )
        rear_axle = ego_state.rear_axle
        x = float(rear_axle.x)
        y = float(rear_axle.y)
        heading = float(rear_axle.heading)
        time_us = int(ego_state.time_point.time_us)
        dt = self.sampling_time
        states = [ego_state]
        points = [(x, y, heading, 0.0)]
        n_steps = max(1, int(round(self.horizon_seconds / dt)))
        accel = (target_speed - current_speed) / max(self.horizon_seconds * 0.5, dt)
        accel = max(-self.max_decel, min(self.max_accel, accel))
        speed = current_speed
        for step in range(1, n_steps + 1):
            speed = max(0.0, min(self.max_speed, speed + accel * dt))
            heading = heading + yaw_rate * dt
            x = x + speed * math.cos(heading) * dt
            y = y + speed * math.sin(heading) * dt
            pose = StateSE2(x, y, heading)
            states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=pose,
                    # nuPlan's dynamic state is rear-axle frame: x=longitudinal speed.
                    rear_axle_velocity_2d=StateVector2D(speed, 0.0),
                    rear_axle_acceleration_2d=StateVector2D(accel, 0.0),
                    tire_steering_angle=0.0,
                    time_point=ego_state.time_point.__class__(
                        time_us + int(step * dt * 1_000_000)
                    ),
                    vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=yaw_rate,
                    angular_accel=0.0,
                )
            )
            points.append((x, y, heading, step * dt))
        return states, points

    def _simulate_corridor_control(
        self,
        ego_state: EgoState,
        target_speed: float,
        yaw_rate: float,
        current_speed: float,
        corridor_path: list[tuple[float, float, float]],
    ) -> tuple[list[EgoState], list[tuple[float, float, float, float]]]:
        rear_axle = ego_state.rear_axle
        time_us = int(ego_state.time_point.time_us)
        dt = self.sampling_time
        states = [ego_state]
        points = [(float(rear_axle.x), float(rear_axle.y), float(rear_axle.heading), 0.0)]
        n_steps = max(1, int(round(self.horizon_seconds / dt)))
        accel = (target_speed - current_speed) / max(self.horizon_seconds * 0.5, dt)
        accel = max(-self.max_decel, min(self.max_accel, accel))
        speed = current_speed
        path_s = self._path_cumulative_distances(corridor_path)
        start_s = self._nearest_progress_on_path(corridor_path, path_s, float(rear_axle.x), float(rear_axle.y))
        travel = 0.0
        for step in range(1, n_steps + 1):
            speed = max(0.0, min(self.max_speed, speed + accel * dt))
            travel += speed * dt
            x, y, path_heading = self._interpolate_path_pose(corridor_path, path_s, start_s + travel)
            # Allow the policy yaw-rate to make only a small local correction; the
            # lane baseline remains the dominant source of lateral control.
            heading = path_heading + 0.15 * yaw_rate
            pose = StateSE2(x, y, heading)
            states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=pose,
                    rear_axle_velocity_2d=StateVector2D(speed, 0.0),
                    rear_axle_acceleration_2d=StateVector2D(accel, 0.0),
                    tire_steering_angle=0.0,
                    time_point=ego_state.time_point.__class__(
                        time_us + int(step * dt * 1_000_000)
                    ),
                    vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=yaw_rate,
                    angular_accel=0.0,
                )
            )
            points.append((x, y, heading, step * dt))
        return states, points

    def _corridor_lead_speed_cap(
        self,
        ego_state: EgoState,
        path: list[tuple[float, float, float]],
        obstacles: list[dict[str, float]],
        current_speed: float,
        nominal_target_speed: float,
    ) -> float | None:
        if not obstacles:
            return None

        path_s = self._path_cumulative_distances(path)
        ego_s = self._nearest_progress_on_path(
            path, path_s, float(ego_state.rear_axle.x), float(ego_state.rear_axle.y)
        )
        speed_caps: list[float] = []
        for obs in obstacles:
            obs_s = self._nearest_progress_on_path(path, path_s, obs["x"], obs["y"])
            obs_x, obs_y, obs_heading = self._interpolate_path_pose(path, path_s, obs_s)
            lateral_dist = math.hypot(obs["x"] - obs_x, obs["y"] - obs_y)
            longitudinal_gap = obs_s - ego_s
            if longitudinal_gap <= 0.0 or longitudinal_gap > 45.0:
                continue
            if lateral_dist > obs["radius"] + 2.5:
                continue

            path_vx = math.cos(obs_heading)
            path_vy = math.sin(obs_heading)
            obs_along_speed = obs["vx"] * path_vx + obs["vy"] * path_vy
            relative_speed = max(current_speed - obs_along_speed, 0.0)
            ttc = longitudinal_gap / max(relative_speed, 0.1)

            desired_gap = self.lead_min_gap + current_speed * self.lead_time_headway
            gap_error = desired_gap - longitudinal_gap
            risk_active = (
                ttc < self.lead_ttc_threshold
                or longitudinal_gap < desired_gap + self.lead_gap_margin
            )
            if not risk_active:
                continue

            # Only cap overly fast candidates. Avoid adding hard-stop candidates
            # that hurt comfort and direction on short closed-loop subsets.
            speed_cap = obs_along_speed + max(0.0, longitudinal_gap - self.lead_min_gap) / max(
                self.lead_time_headway, 0.5
            )
            if gap_error > 0.0:
                speed_cap = min(
                    speed_cap,
                    current_speed - min(self.max_comfortable_decel, gap_error * 0.15),
                )
            if ttc < max(2.0, 0.75 * self.lead_ttc_threshold):
                speed_cap = min(speed_cap, obs_along_speed + 1.0)
            speed_floor = max(0.0, current_speed - self.lead_max_speed_drop)
            speed_caps.append(min(nominal_target_speed, max(speed_floor, speed_cap)))

        if not speed_caps:
            return None
        return max(0.0, min(self.max_speed, min(speed_caps)))

    @staticmethod
    def _path_cumulative_distances(path: list[tuple[float, float, float]]) -> list[float]:
        distances = [0.0]
        for prev, curr in zip(path, path[1:]):
            distances.append(distances[-1] + math.hypot(curr[0] - prev[0], curr[1] - prev[1]))
        return distances

    @staticmethod
    def _nearest_progress_on_path(
        path: list[tuple[float, float, float]],
        path_s: list[float],
        x: float,
        y: float,
    ) -> float:
        best_s = 0.0
        best_dist = float("inf")
        for idx in range(len(path) - 1):
            x1, y1, _ = path[idx]
            x2, y2, _ = path[idx + 1]
            vx = x2 - x1
            vy = y2 - y1
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-6:
                continue
            ratio = max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / length_sq))
            px = x1 + ratio * vx
            py = y1 + ratio * vy
            dist = math.hypot(x - px, y - py)
            if dist < best_dist:
                best_dist = dist
                best_s = path_s[idx] + ratio * math.sqrt(length_sq)
        return best_s

    @staticmethod
    def _interpolate_path_pose(
        path: list[tuple[float, float, float]],
        path_s: list[float],
        target_s: float,
    ) -> tuple[float, float, float]:
        if target_s <= 0.0:
            return path[0]
        if target_s >= path_s[-1]:
            return path[-1]
        for idx in range(len(path_s) - 1):
            if path_s[idx] <= target_s <= path_s[idx + 1]:
                span = max(path_s[idx + 1] - path_s[idx], 1e-6)
                ratio = (target_s - path_s[idx]) / span
                x1, y1, h1 = path[idx]
                x2, y2, _ = path[idx + 1]
                heading = math.atan2(y2 - y1, x2 - x1) if span > 1e-6 else h1
                return x1 + ratio * (x2 - x1), y1 + ratio * (y2 - y1), heading
        return path[-1]

    @staticmethod
    def _unique_floats(values: list[float], decimals: int = 3) -> list[float]:
        seen = set()
        out = []
        for value in values:
            key = round(float(value), decimals)
            if key not in seen:
                seen.add(key)
                out.append(float(value))
        return out

    def _extract_safety_obstacles(
        self,
        observation: Any | None,
        ego_state: EgoState,
    ) -> list[dict[str, float]]:
        tracked = getattr(observation, "tracked_objects", observation)
        items = getattr(tracked, "tracked_objects", tracked)
        if items is None:
            return []

        ego_x = float(ego_state.center.x)
        ego_y = float(ego_state.center.y)
        vehicle = ego_state.car_footprint.vehicle_parameters
        ego_radius = 0.5 * math.hypot(float(vehicle.length), float(vehicle.width))
        obstacles: list[dict[str, float]] = []
        for obj in list(items):
            box = getattr(obj, "box", obj)
            center = getattr(box, "center", None)
            if center is None:
                continue
            x = float(getattr(center, "x", 0.0))
            y = float(getattr(center, "y", 0.0))
            if math.hypot(x - ego_x, y - ego_y) > 80.0:
                continue
            velocity = getattr(obj, "velocity", None)
            vx = float(getattr(velocity, "x", 0.0)) if velocity is not None else 0.0
            vy = float(getattr(velocity, "y", 0.0)) if velocity is not None else 0.0
            length = float(getattr(box, "length", 4.2))
            width = float(getattr(box, "width", 1.8))
            obstacles.append({
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "radius": ego_radius + 0.5 * math.hypot(length, width) + 0.8,
            })
        return obstacles

    def _safety_score(
        self,
        points: list[tuple[float, float, float, float]],
        obstacles: list[dict[str, float]],
        map_api: Any | None,
        nominal_target_speed: float,
        nominal_yaw_rate: float,
        target_speed: float,
        yaw_rate: float,
        current_speed: float,
        previous_accel: float,
        previous_yaw_rate: float,
        lane_refs: list[tuple[float, float]],
    ) -> float:
        collision_cost = 0.0
        ttc_cost = 0.0
        proximity_cost = 0.0
        for x, y, _, t in points[1:]:
            for obs in obstacles:
                ox = obs["x"] + obs["vx"] * t
                oy = obs["y"] + obs["vy"] * t
                clearance = math.hypot(x - ox, y - oy) - obs["radius"]
                if clearance < 0.0:
                    collision_cost += 1.0 + 2.0 * abs(clearance)
                    if t <= 4.0:
                        ttc_cost += 1.0 + (4.0 - t) / 4.0
                elif self.ttc_proxy and t <= self.ttc_proxy_horizon and clearance < self.ttc_proxy_clearance:
                    time_weight = (self.ttc_proxy_horizon - t) / max(self.ttc_proxy_horizon, 1e-3)
                    clearance_weight = (self.ttc_proxy_clearance - clearance) / max(
                        self.ttc_proxy_clearance,
                        1e-3,
                    )
                    ttc_cost += time_weight * clearance_weight**2
                elif clearance < 6.0:
                    proximity_cost += ((6.0 - clearance) / 6.0) ** 2

        drivable_cost = self._drivable_cost(points, map_api)
        final_x, final_y, _, _ = points[-1]
        start_x, start_y, _, _ = points[0]
        progress = math.hypot(final_x - start_x, final_y - start_y)
        nominal_cost = 0.2 * abs(target_speed - nominal_target_speed)
        nominal_cost += 0.5 * abs(yaw_rate - nominal_yaw_rate)
        accel = (target_speed - current_speed) / max(self.horizon_seconds * 0.5, self.sampling_time)
        smoothness_cost = self._smoothness_cost(accel, previous_accel, yaw_rate, previous_yaw_rate)
        lane_center_cost = self._lane_center_cost(points, lane_refs)
        return (
            self.collision_weight * collision_cost
            + self.ttc_weight * ttc_cost
            + 2.0 * proximity_cost
            + self.drivable_weight * drivable_cost
            + self.smoothness_weight * smoothness_cost
            + self.lane_center_weight * lane_center_cost
            + nominal_cost
            - self.progress_weight * progress
        )

    def _smoothness_cost(
        self,
        accel: float,
        previous_accel: float,
        yaw_rate: float,
        previous_yaw_rate: float,
    ) -> float:
        decel_excess = max(0.0, -accel - self.max_comfortable_decel)
        jerk = abs(accel - previous_accel) / max(self.sampling_time, 1e-3)
        jerk_excess = max(0.0, jerk - self.max_comfortable_jerk)
        yaw_change = abs(yaw_rate - previous_yaw_rate)
        return decel_excess**2 + 0.25 * jerk_excess**2 + 0.5 * yaw_change**2

    def _extract_lane_center_refs(
        self,
        ego_state: EgoState,
        map_api: Any | None,
        route_roadblock_ids: list[str] | None,
    ) -> list[tuple[float, float]]:
        if map_api is None or Point2D is None or SemanticMapLayer is None:
            return []
        try:
            nearby = map_api.get_proximal_map_objects(
                Point2D(float(ego_state.center.x), float(ego_state.center.y)),
                60.0,
                [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            )
        except Exception:
            return []

        route_ids = set(route_roadblock_ids or [])
        route_refs: list[tuple[float, float]] = []
        fallback_refs: list[tuple[float, float]] = []
        for layer_items in nearby.values():
            for item in list(layer_items):
                points = self._baseline_points(item)
                if not points:
                    continue
                fallback_refs.extend(points[:: max(1, len(points) // 12)])
                if route_ids and self._is_on_route(item, route_ids):
                    route_refs.extend(points[:: max(1, len(points) // 12)])
        refs = route_refs if route_refs else fallback_refs
        return refs[:200]

    @staticmethod
    def _baseline_points(map_object: Any) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in DoorRLNuPlanPlanner._baseline_path(map_object)]

    @staticmethod
    def _baseline_path(map_object: Any) -> list[tuple[float, float, float]]:
        baseline = getattr(map_object, "baseline_path", None)
        discrete = getattr(baseline, "discrete_path", None)
        if not discrete:
            return []
        return [
            (
                float(getattr(pose, "x", 0.0)),
                float(getattr(pose, "y", 0.0)),
                float(getattr(pose, "heading", 0.0)),
            )
            for pose in discrete
        ]

    def _extract_lane_corridors(
        self,
        ego_state: EgoState,
        map_api: Any | None,
        route_roadblock_ids: list[str] | None,
    ) -> list[list[tuple[float, float, float]]]:
        if map_api is None or Point2D is None or SemanticMapLayer is None:
            return []
        try:
            nearby = map_api.get_proximal_map_objects(
                Point2D(float(ego_state.center.x), float(ego_state.center.y)),
                80.0,
                [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            )
        except Exception:
            return []

        route_ids = set(route_roadblock_ids or [])
        route_paths: list[list[tuple[float, float, float]]] = []
        fallback_paths: list[list[tuple[float, float, float]]] = []
        for layer_items in nearby.values():
            for item in list(layer_items):
                path = self._baseline_path(item)
                if len(path) < 2:
                    continue
                if route_ids and self._is_on_route(item, route_ids):
                    route_paths.append(path)
                else:
                    fallback_paths.append(path)

        paths = route_paths if route_paths else fallback_paths
        paths.sort(key=lambda p: self._corridor_sort_key(ego_state, p))
        return paths[:12]

    @staticmethod
    def _corridor_sort_key(
        ego_state: EgoState,
        path: list[tuple[float, float, float]],
    ) -> float:
        x = float(ego_state.rear_axle.x)
        y = float(ego_state.rear_axle.y)
        heading = float(ego_state.rear_axle.heading)
        distances = DoorRLNuPlanPlanner._path_cumulative_distances(path)
        start_s = DoorRLNuPlanPlanner._nearest_progress_on_path(path, distances, x, y)
        px, py, pheading = DoorRLNuPlanPlanner._interpolate_path_pose(path, distances, start_s)
        lateral_dist = math.hypot(x - px, y - py)
        heading_error = abs(math.atan2(math.sin(heading - pheading), math.cos(heading - pheading)))
        remaining = max(0.0, distances[-1] - start_s)
        return lateral_dist + 2.0 * heading_error - 0.02 * remaining

    @staticmethod
    def _is_on_route(map_object: Any, route_ids: set[str]) -> bool:
        get_roadblock = getattr(map_object, "get_roadblock", None)
        roadblock = get_roadblock() if callable(get_roadblock) else None
        get_roadblock_id = getattr(map_object, "get_roadblock_id", None)
        roadblock_id = get_roadblock_id() if callable(get_roadblock_id) else None
        candidates = [
            getattr(map_object, "id", None),
            getattr(getattr(map_object, "parent", None), "id", None),
            getattr(roadblock, "id", None),
            roadblock_id,
        ]
        return any(str(candidate) in route_ids for candidate in candidates if candidate is not None)

    @staticmethod
    def _lane_center_cost(
        points: list[tuple[float, float, float, float]],
        lane_refs: list[tuple[float, float]],
    ) -> float:
        if not lane_refs:
            return 0.0
        costs = []
        for index, (x, y, _, _) in enumerate(points[1:], start=1):
            if index % 4 != 0 and index != len(points) - 1:
                continue
            min_dist = min(math.hypot(x - lx, y - ly) for lx, ly in lane_refs)
            costs.append(min((min_dist / 3.0) ** 2, 4.0))
        return sum(costs) / max(len(costs), 1)

    @staticmethod
    def _drivable_cost(
        points: list[tuple[float, float, float, float]],
        map_api: Any | None,
    ) -> float:
        if (
            map_api is None
            or Point2D is None
            or SemanticMapLayer is None
        ):
            return 0.0
        checked = 0
        off_drivable = 0
        for index, (x, y, _, _) in enumerate(points[1:], start=1):
            if index % 4 != 0 and index != len(points) - 1:
                continue
            checked += 1
            try:
                if not map_api.is_in_layer(Point2D(x, y), SemanticMapLayer.DRIVABLE_AREA):
                    off_drivable += 1
            except Exception:
                return 0.0
        return off_drivable / max(checked, 1)


def add_nuplan_devkit_to_path(devkit_root: str | Path) -> None:
    """Convenience helper for scripts launched from the DOOR-RL repo."""
    root = str(Path(devkit_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
