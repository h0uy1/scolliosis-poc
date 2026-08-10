import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

import cobb_evaluate
import decoder
import draw_points
import pre_proc
from models import spinal_net


class SpineInferenceService:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_h: int = 1024,
        input_w: int = 512,
        down_ratio: int = 4,
        conf_thresh: float = 0.2,
        num_classes: int = 1,
        device: Optional[str] = None,
    ):
        self.root_dir = Path(__file__).resolve().parent
        self.weights_path = Path(
            weights_path
            or os.getenv("MODEL_WEIGHTS")
            or self.root_dir / "weights_spinal" / "model_last.pth"
        )
        self.input_h = int(os.getenv("MODEL_INPUT_H", input_h))
        self.input_w = int(os.getenv("MODEL_INPUT_W", input_w))
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.device = self._select_device(device)
        self.decoder = decoder.DecDecoder(K=17, conf_thresh=conf_thresh)
        self.model = self._load_model()

    def _select_device(self, requested_device: Optional[str]) -> torch.device:
        selected = requested_device or os.getenv("MODEL_DEVICE")
        if selected:
            selected = selected.lower()
            if selected == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(selected)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> torch.nn.Module:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        heads = {
            "hm": self.num_classes,
            "reg": 2 * self.num_classes,
            "wh": 2 * 4,
        }
        model = spinal_net.SpineNet(
            heads=heads,
            pretrained=False,
            down_ratio=self.down_ratio,
            final_kernel=1,
            head_conv=256,
        )

        checkpoint = torch.load(
            str(self.weights_path),
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes, include_overlay: bool = True) -> Dict[str, Any]:
        image = self._decode_image(image_bytes)
        original_h, original_w = image.shape[:2]

        image_tensor = pre_proc.processing_test(
            image=image,
            input_h=self.input_h,
            input_w=self.input_w,
        ).to(self.device)

        with torch.inference_mode():
            output = self.model(image_tensor)
            pts = self.decoder.ctdet_decode(output["hm"], output["wh"], output["reg"])

        pts[:, :10] *= self.down_ratio
        pts = pts[np.argsort(pts[:, 1])]

        vertebrae = self._build_vertebrae(pts)
        landmarks_model = self._landmarks_from_vertebrae(vertebrae, scaled_to_original=False)
        landmarks_original = self._landmarks_from_vertebrae(vertebrae, scaled_to_original=True,
                                                            original_w=original_w,
                                                            original_h=original_h)

        angles = {"main": None, "upper": None, "lower": None}
        overlay_base64 = None

        if landmarks_model:
            resized_image = cv2.resize(image, (self.input_w, self.input_h))
            overlay_image = resized_image.copy()
            angle_values = cobb_evaluate.draw_cobb_overlay(
                np.asarray(landmarks_model, np.float32),
                overlay_image,
            )
            angle_names = ["main", "upper", "lower"]
            angles = {
                name: round(float(angle_values[idx]), 2) if idx < len(angle_values) else None
                for idx, name in enumerate(angle_names)
            }

            if include_overlay:
                regress_image = resized_image.copy()
                points_image = resized_image.copy()
                regress_image, _ = draw_points.draw_landmarks_regress_test(
                    pts,
                    regress_image,
                    points_image,
                )
                cobb_evaluate.draw_cobb_overlay(np.asarray(landmarks_model, np.float32), regress_image)
                overlay_base64 = self._encode_image(regress_image)

        return {
            "cobb_angles": angles,
            "landmarks": landmarks_original,
            "landmarks_model_input": landmarks_model,
            "vertebrae": vertebrae,
            "overlay_image_base64": overlay_base64,
            "overlay_image_mime_type": "image/jpeg" if overlay_base64 else None,
            "input_size": {
                "width": self.input_w,
                "height": self.input_h,
            },
            "original_size": {
                "width": original_w,
                "height": original_h,
            },
            "device": str(self.device),
        }

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Empty image upload")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Upload is not a readable image")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    def _build_vertebrae(self, pts: np.ndarray) -> List[Dict[str, Any]]:
        vertebrae = []
        for index, pt in enumerate(pts):
            corners = [
                [float(pt[2]), float(pt[3])],
                [float(pt[4]), float(pt[5])],
                [float(pt[6]), float(pt[7])],
                [float(pt[8]), float(pt[9])],
            ]
            vertebrae.append(
                {
                    "index": index + 1,
                    "center": [float(pt[0]), float(pt[1])],
                    "corners": corners,
                    "score": float(pt[10]),
                }
            )
        return vertebrae

    def _landmarks_from_vertebrae(
        self,
        vertebrae: List[Dict[str, Any]],
        scaled_to_original: bool,
        original_w: Optional[int] = None,
        original_h: Optional[int] = None,
    ) -> List[List[float]]:
        landmarks = []
        scale_x = float(original_w) / self.input_w if scaled_to_original and original_w else 1.0
        scale_y = float(original_h) / self.input_h if scaled_to_original and original_h else 1.0

        for vertebra in vertebrae:
            for x, y in vertebra["corners"]:
                landmarks.append([round(x * scale_x, 2), round(y * scale_y, 2)])
        return landmarks

    def _encode_image(self, image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Failed to encode overlay image")
        return base64.b64encode(encoded.tobytes()).decode("ascii")


_SERVICE: Optional[SpineInferenceService] = None


def get_inference_service() -> SpineInferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SpineInferenceService()
    return _SERVICE
