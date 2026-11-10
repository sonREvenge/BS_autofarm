from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)


def image_input(model):
    inputs = model.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Expected one image input")
    image = inputs[0]
    shape = image.shape
    if (
        image.type != "tensor(float)"
        or len(shape) != 4
        or shape[:2] != [1, 3]
        or not all(type(v) is int and v > 0 for v in shape[2:])
    ):
        raise ValueError(f"Expected fixed float32 [1,3,H,W] input, got {shape} ({image.type})")
    return image, tuple(shape[2:])


@dataclass(frozen=True)
class Transform:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    def points(self, points: np.ndarray) -> np.ndarray:
        result = points.astype(np.float32, copy=True)
        result[..., 0] = (result[..., 0] - self.left) * self.width / self.resized_width
        result[..., 1] = (result[..., 1] - self.top) * self.height / self.resized_height
        result[..., 0] = result[..., 0].clip(0, self.width - 1)
        result[..., 1] = result[..., 1].clip(0, self.height - 1)
        return result


def letterbox(rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, Transform]:
    height, width = rgb.shape[:2]
    target_h, target_w = size
    ratio = min(target_w / width, target_h / height)
    rw, rh = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (target_w - rw) // 2, (target_h - rh) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top : top + rh, left : left + rw] = cv2.resize(rgb, (rw, rh))
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor *= 1 / 255
    return tensor, Transform(width, height, rw, rh, left, top)


def _open_session(path: Path, options, providers) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as error:
        raise RuntimeError(f"Cannot load {path.name}: {error}") from error


def session(path: Path, provider: str) -> ort.InferenceSession:
    names = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
    }
    available = ort.get_available_providers()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    if provider == "auto":
        preferred = [p for p in (names["cuda"], names["directml"]) if p in available]
    else:
        preferred = [names[provider]]
        if preferred[0] not in available:
            raise ValueError(f"{provider} unavailable; installed providers: {available}")
    providers = list(dict.fromkeys(preferred + [names["cpu"]]))
    try:
        result = _open_session(path, options, providers)
    except RuntimeError:
        if provider != "auto" or providers == [names["cpu"]]:
            raise
        log.warning("Accelerator failed to load; using CPU", exc_info=True)
        result = _open_session(path, options, [names["cpu"]])
    if provider != "auto" and result.get_providers()[0] != names[provider]:
        raise ValueError(f"Requested {provider}, but ONNX Runtime fell back to CPU")
    log.info("%s: %s", path.name, result.get_providers()[0])
    return result


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.6) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        overlap = np.maximum(
            0, np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2])
        )
        intersection = overlap.prod(axis=1)
        iou = intersection / np.maximum(areas[i] + areas[rest] - intersection, 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode(
    raw: np.ndarray, classes: int, keypoints: tuple[int, int], end_to_end: bool, threshold: float
) -> np.ndarray:
    channels = (6 if end_to_end else 4 + classes) + keypoints[0] * keypoints[1]
    rows = np.asarray(raw)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"Unsupported entity output: {rows.shape}")
    if rows.shape[1] != channels and rows.shape[0] == channels:
        rows = rows.T
    if rows.shape[1] != channels:
        raise ValueError(f"Expected {channels} channels, got {rows.shape}")
    rows = rows[np.isfinite(rows).all(axis=1)]
    if end_to_end:
        result = rows[rows[:, 4] >= threshold].copy()
    else:
        scores = rows[:, 4 : 4 + classes]
        ids = scores.argmax(axis=1)
        confidence = scores[np.arange(len(rows)), ids]
        mask = confidence >= threshold
        rows, ids, confidence = rows[mask], ids[mask], confidence[mask]
        xyxy = np.column_stack((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2))
        result = np.column_stack((xyxy, confidence, ids, rows[:, 4 + classes :]))
    valid = (
        (result[:, 2] > result[:, 0])
        & (result[:, 3] > result[:, 1])
        & (result[:, 5] >= 0)
        & (result[:, 5] < classes)
        & (result[:, 5] == np.floor(result[:, 5]))
        & (result[:, 4] <= 1)
    )
    result = result[valid]
    if not end_to_end:
        keep = []
        for cls in np.unique(result[:, 5]):
            indices = np.flatnonzero(result[:, 5] == cls)
            keep.extend(indices[nms(result[indices, :4], result[indices, 4])])
        result = result[keep]
    return result[np.argsort(-result[:, 4], kind="stable")]


@dataclass(frozen=True)
class Entity:
    name: str
    score: float
    box: tuple[float, float, float, float]
    ground: tuple[float, float]


class OnnxModel:
    def __init__(self, path: Path, provider: str):
        self.path = path
        self.provider = provider
        self.model = session(path, provider)
        self.input, self.size = image_input(self.model)

    def _execute(self, outputs, tensor):
        try:
            return self.model.run(outputs, {self.input.name: tensor})
        except Exception as error:
            raise RuntimeError(f"Inference failed for {self.path.name}: {error}") from error

    def run(self, outputs, tensor):
        try:
            return self._execute(outputs, tensor)
        except RuntimeError:
            if self.provider != "auto" or self.model.get_providers()[0] == "CPUExecutionProvider":
                raise
            log.warning(
                "%s: accelerator inference failed; retrying on CPU", self.path.name, exc_info=True
            )
            self.model = session(self.path, "cpu")
            self.input, self.size = image_input(self.model)
            return self._execute(outputs, tensor)


class EntityDetector(OnnxModel):
    def __init__(self, path: Path, provider: str):
        super().__init__(path, provider)
        metadata = self.model.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(metadata["names"])
        self.names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
        if not self.names or not all(isinstance(name, str) for name in self.names):
            raise ValueError("Entity model has no valid class names")
        self.keypoints = tuple(ast.literal_eval(metadata.get("kpt_shape", "[0, 0]")))
        if self.keypoints not in ((0, 0), (1, 2), (1, 3)):
            raise ValueError(f"Unsupported keypoints: {self.keypoints}")
        self.end_to_end = metadata.get("end2end", "false").lower() == "true"
        if len(self.model.get_outputs()) != 1:
            raise ValueError("Expected one entity prediction output")

    def detect(self, rgb: np.ndarray, threshold: float) -> list[Entity]:
        tensor, transform = letterbox(rgb, self.size)
        raw = self.run(None, tensor)[0]
        rows = decode(raw, len(self.names), self.keypoints, self.end_to_end, threshold)
        result = []
        for row in rows:
            box = transform.points(row[:4].reshape(2, 2)).ravel()
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            ground = ((box[0] + box[2]) / 2, box[3])
            if self.keypoints[0] and (self.keypoints[1] == 2 or row[8] >= 0.5):
                ground = tuple(transform.points(row[6:8]))
            result.append(Entity(self.names[int(row[5])], float(row[4]), tuple(box), ground))
        return result


@dataclass(frozen=True)
class Terrain:
    movement: np.ndarray
    projectiles: np.ndarray


class TerrainDetector(OnnxModel):
    MOVEMENT = {"wall", "rope_fence", "water", "out_of_bounds", "indestructible_wall", "fragile"}
    PROJECTILES = {"wall", "out_of_bounds", "indestructible_wall", "fragile"}

    def __init__(self, path: Path, provider: str):
        super().__init__(path, provider)
        names = [o.name for o in self.model.get_outputs()]
        if set(names) != {"class_ids", "confidence"}:
            raise ValueError("Expected terrain class_ids and confidence outputs")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.classes = {int(key): value for key, value in metadata["class_ids"].items()}
        if not self.MOVEMENT <= set(self.classes.values()):
            raise ValueError("Terrain model is missing required obstacle classes")

    def detect(self, rgb: np.ndarray, threshold: float) -> Terrain:
        tensor, transform = letterbox(rgb, self.size)
        ids, confidence = self.run(["class_ids", "confidence"], tensor)
        if ids.shape != (1, *self.size) or confidence.shape != ids.shape:
            raise ValueError("Terrain output does not match its input dimensions")
        t = transform
        crop = (0, slice(t.top, t.top + t.resized_height), slice(t.left, t.left + t.resized_width))
        ids, confidence = ids[crop], confidence[crop]
        movement = np.zeros(ids.shape, dtype="uint8")
        projectiles = np.zeros_like(movement)
        for class_id, name in self.classes.items():
            if name not in self.MOVEMENT:
                continue
            minimum = (
                0
                if name == "out_of_bounds"
                else max(threshold, {"rope_fence": 0.65, "water": 0.75}.get(name, 0.6))
            )
            region = ((ids == class_id) & np.isfinite(confidence) & (confidence >= minimum)).astype(
                "uint8"
            )
            if name != "out_of_bounds" and region.any():
                count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
                area = stats[:, cv2.CC_STAT_AREA]
                mean = np.bincount(labels.ravel(), weights=confidence.ravel(), minlength=count)
                mean /= np.maximum(area, 1)
                keep = (area >= 0.0006 * t.resized_width * t.resized_height) & (mean >= 0.8)
                keep[0] = False
                region = keep[labels].astype("uint8")
            movement |= region
            if name in self.PROJECTILES:
                projectiles |= region

        def restore(mask):
            return cv2.resize(mask, (t.width, t.height), interpolation=cv2.INTER_NEAREST).astype(
                bool
            )

        return Terrain(restore(movement), restore(projectiles))
