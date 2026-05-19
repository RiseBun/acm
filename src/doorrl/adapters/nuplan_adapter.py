from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)


class NuPlanClosedLoopAdapter:
    def __init__(self, spec: TokenizationSpec, reactive: bool = True) -> None:
        self.spec = spec
        self.reactive = reactive
        self.converter = NormalizedSceneConverter(spec)

    @property
    def mode(self) -> BenchmarkMode:
        return (
            BenchmarkMode.CLOSED_LOOP_REACTIVE
            if self.reactive
            else BenchmarkMode.CLOSED_LOOP_NON_REACTIVE
        )

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="nuplan",
            mode=self.mode,
            purpose="Primary closed-loop benchmark for reactive vs non-reactive driving evaluation.",
            expected_inputs=[
                "planner observation",
                "tracked objects",
                "map context",
                "ego command or trajectory target",
            ],
            outputs=[
                "oracle scene tokens",
                "closed-loop metrics",
                "reactive/non-reactive experiment tags",
            ],
        )

    def build_scene_item_from_normalized(
        self,
        normalized_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self.converter.build_scene_item(normalized_record)

    def supported_experiments(self) -> Dict[str, str]:
        return {
            "replay_train_replay_test": "Train and test in non-reactive mode.",
            "replay_train_reactive_test": "Train in non-reactive mode, test in reactive mode.",
            "reactive_train_reactive_test": "Train and test in reactive mode.",
        }

    def convert_nuplan_observation(self, observation: Any) -> Dict[str, Any]:
        ego_state = self._extract_ego_state(observation)
        objects = self._extract_objects(observation)
        relations = self._compute_relations(ego_state, objects)
        return self.build_scene_item_from_normalized({
            "ego": ego_state,
            "objects": objects,
            "map_elements": [],
            "relations": relations,
            "action": [0.0, 0.0],
            "reward": 0.0,
            "continue": 1.0,
        })

    def convert_planner_input(
        self,
        current_input: Any,
        initialization: Any | None = None,
    ) -> Dict[str, Any]:
        """Convert nuPlan ``PlannerInput`` into a DOOR-RL scene item.

        This method intentionally uses duck typing so importing this package
        does not require ``nuplan-devkit``. With the devkit installed, it reads
        the latest ego state and ``DetectionsTracks`` observation from the
        planner history, then optionally queries nearby lanes from
        ``PlannerInitialization.map_api``.
        """
        history = getattr(current_input, "history", None)
        ego_state = self._latest_from_sequence(getattr(history, "ego_states", None))
        observation = self._latest_from_sequence(getattr(history, "observations", None))
        ego = self._extract_ego_state(ego_state)
        objects = self._extract_objects(observation)
        map_elements = self._extract_map_elements(ego, initialization)
        relations = self._compute_relations(ego, objects)
        return self.build_scene_item_from_normalized({
            "ego": ego,
            "objects": objects,
            "map_elements": map_elements,
            "relations": relations,
            "action": [0.0, 0.0],
            "reward": 0.0,
            "continue": 1.0,
        })

    @staticmethod
    def _latest_from_sequence(value: Any) -> Any:
        if value is None:
            return None
        try:
            return value[-1]
        except Exception:
            return value

    @staticmethod
    def _xy_from_pose(pose: Any) -> tuple[float, float, float]:
        if pose is None:
            return 0.0, 0.0, 0.0
        x = float(getattr(pose, "x", 0.0))
        y = float(getattr(pose, "y", 0.0))
        heading = float(getattr(pose, "heading", 0.0))
        return x, y, heading

    @staticmethod
    def _vector_xy(vector: Any) -> tuple[float, float]:
        if vector is None:
            return 0.0, 0.0
        return float(getattr(vector, "x", 0.0)), float(getattr(vector, "y", 0.0))

    def _extract_ego_state(self, ego_state: Any) -> Dict[str, float]:
        center = getattr(ego_state, "center", None)
        if center is None:
            footprint = getattr(ego_state, "car_footprint", None)
            center = getattr(footprint, "center", None)
        x, y, heading = self._xy_from_pose(center)

        dynamic = getattr(ego_state, "dynamic_car_state", None)
        velocity = getattr(dynamic, "center_velocity_2d", None)
        if velocity is None:
            velocity = getattr(dynamic, "rear_axle_velocity_2d", None)
        vx, vy = self._vector_xy(velocity)

        footprint = getattr(ego_state, "car_footprint", None)
        vehicle = getattr(footprint, "vehicle_parameters", None)
        length = float(getattr(vehicle, "length", 4.5))
        width = float(getattr(vehicle, "width", 1.8))
        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "heading": heading,
            "length": length,
            "width": width,
            "speed": math.hypot(vx, vy),
            "visibility": 1.0,
        }

    def _extract_objects(self, observation: Any) -> list[Dict[str, Any]]:
        tracked = getattr(observation, "tracked_objects", observation)
        items = getattr(tracked, "tracked_objects", tracked)
        if items is None:
            return []
        objects = []
        for obj in list(items)[: self.spec.max_dynamic_objects]:
            box = getattr(obj, "box", obj)
            center = getattr(box, "center", None)
            x, y, heading = self._xy_from_pose(center)
            velocity = getattr(obj, "velocity", None)
            vx, vy = self._vector_xy(velocity)
            object_type = getattr(obj, "tracked_object_type", "")
            token_type = self._token_type_to_str(object_type)
            objects.append({
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "heading": heading,
                "length": float(getattr(box, "length", 4.2)),
                "width": float(getattr(box, "width", 1.8)),
                "token_type": token_type,
                "speed": math.hypot(vx, vy),
                "visibility": 1.0,
            })
        return objects

    def _extract_map_elements(
        self,
        ego: Mapping[str, float],
        initialization: Any | None,
    ) -> list[Dict[str, float]]:
        if initialization is None:
            return []
        map_api = getattr(initialization, "map_api", None)
        if map_api is None:
            return []
        try:
            from nuplan.common.actor_state.state_representation import Point2D
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer

            layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            nearby = map_api.get_proximal_map_objects(
                Point2D(float(ego["x"]), float(ego["y"])),
                40.0,
                layers,
            )
        except Exception:
            return []

        elements: list[Dict[str, float]] = []
        for layer_items in nearby.values():
            for item in list(layer_items):
                baseline = getattr(item, "baseline_path", None)
                discrete = getattr(baseline, "discrete_path", None)
                if not discrete:
                    continue
                pose = discrete[0]
                x, y, heading = self._xy_from_pose(pose)
                elements.append({
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "visibility": 1.0,
                    "priority": 1.0,
                })
                if len(elements) >= self.spec.max_map_tokens:
                    return elements
        return elements

    def _compute_relations(
        self,
        ego: Mapping[str, float],
        objects: list[Dict[str, Any]],
    ) -> list[Dict[str, float]]:
        relations = []
        ego_x = float(ego.get("x", 0.0))
        ego_y = float(ego.get("y", 0.0))
        ego_vx = float(ego.get("vx", 0.0))
        ego_vy = float(ego.get("vy", 0.0))
        for obj in objects:
            dx = float(obj.get("x", 0.0)) - ego_x
            dy = float(obj.get("y", 0.0)) - ego_y
            rel_vx = float(obj.get("vx", 0.0)) - ego_vx
            rel_vy = float(obj.get("vy", 0.0)) - ego_vy
            distance = math.hypot(dx, dy)
            risk = 1.0 / max(distance, 1.0)
            lane_conflict = 1.0 if abs(dy) < 2.0 else 0.0
            relations.append({
                "x": dx,
                "y": dy,
                "vx": rel_vx,
                "vy": rel_vy,
                "distance": distance,
                "ttc": self._compute_ttc(dx, dy, rel_vx, rel_vy),
                "risk": risk,
                "lane_conflict": lane_conflict,
                "visibility": 1.0,
                "priority": self._priority(obj, distance, lane_conflict),
                "is_interactive": 1.0 if distance < 20.0 or lane_conflict else 0.0,
            })
        relations.sort(key=lambda r: r["risk"], reverse=True)
        return relations[: self.spec.max_relation_tokens]

    @staticmethod
    def _compute_ttc(dx: float, dy: float, rel_vx: float, rel_vy: float) -> float:
        distance = math.hypot(dx, dy)
        closing = (dx * rel_vx + dy * rel_vy) / max(distance, 0.1)
        if closing > 0:
            return min(distance / closing, 20.0)
        return 20.0

    @staticmethod
    def _priority(obj: Mapping[str, Any], distance: float, lane_conflict: float) -> float:
        value = 0.5
        if obj.get("token_type") in {"pedestrian", "cyclist"}:
            value += 0.2
        if distance < 10.0:
            value += 0.2
        elif distance < 20.0:
            value += 0.1
        if lane_conflict:
            value += 0.1
        return min(value, 1.0)

    @staticmethod
    def _token_type_to_str(object_type: Any) -> str:
        text = str(object_type).lower()
        if "pedestrian" in text:
            return "pedestrian"
        if "bicycle" in text or "cyclist" in text:
            return "cyclist"
        return "vehicle"
