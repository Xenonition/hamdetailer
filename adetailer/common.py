from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from rich import print  # noqa: A004  Shadowing built-in 'print'
from torchvision.transforms.functional import to_pil_image

REPO_ID = "Bingsu/adetailer"

T = TypeVar("T", int, float)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download(file: str, repo_id: str = REPO_ID, check_remote: bool = True) -> str:
    if check_remote:
        with suppress(Exception):
            return hf_hub_download(repo_id, file, etag_timeout=1)

        with suppress(Exception):
            return hf_hub_download(
                repo_id, file, etag_timeout=1, endpoint="https://hf-mirror.com"
            )

    with suppress(Exception):
        return hf_hub_download(repo_id, file, local_files_only=True)

    if check_remote:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        print(msg)
    return "INVALID"


def safe_mkdir(path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists() and os.access(path.parent, os.W_OK):
        path.mkdir()


def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


def download_models(*names: str, check_remote: bool = True) -> dict[str, str]:
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    check_remote=check_remote,
                )
            else:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    check_remote=check_remote,
                )
    return {name: future.result() for name, future in models.items()}


def get_models(
    *dirs: str | os.PathLike[str], huggingface: bool = True
) -> OrderedDict[str, str]:
    model_paths = []

    for dir_ in dirs:
        if not dir_:
            continue
        model_paths.extend(scan_model_dir(Path(dir_)))

    models = OrderedDict()
    to_download = [
        "face_yolov8n.pt",
        "face_yolov8s.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "yolov8x-worldv2.pt",
    ]
    models.update(download_models(*to_download, check_remote=huggingface))

    models.update(
        {
            "mediapipe_face_full": "mediapipe_face_full",
            "mediapipe_face_short": "mediapipe_face_short",
            "mediapipe_face_mesh": "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
        }
    )

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)  # noqa: PLW2901
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image


def color_for_label(label: str) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (244, 162, 97),
        (233, 196, 106),
        (38, 70, 83),
        (142, 68, 173),
        (39, 174, 96),
        (241, 196, 15),
        (52, 152, 219),
        (231, 76, 60),
    ]
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(palette)
    return palette[idx]


def draw_detection_overlay(
    image: Image.Image,
    bboxes: list[list[float]] | list[list[int]],
    masks: list[Image.Image] | None = None,
    confidences: list[float] | None = None,
    labels: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
) -> Image.Image:
    base = image.copy()
    masks = masks or []

    font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        color = None
        if colors and i < len(colors):
            color = colors[i]
        elif labels and i < len(labels):
            color = color_for_label(labels[i])
        else:
            color = (230, 57, 70)

        if i < len(masks):
            overlay = Image.new("RGB", base.size, color)
            masked = Image.composite(overlay, base, masks[i])
            base = Image.blend(base, masked, 0.25)

        draw = ImageDraw.Draw(base)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = labels[i] if labels and i < len(labels) else None
        if confidences and i < len(confidences):
            score_text = f"{float(confidences[i]):.2f}"
            label = f"{label} {score_text}" if label else f"det {score_text}"
        if label:
            text_pos = (min(x2 + 4, base.size[0] - 2), y1 + 2)
            draw.text(text_pos, label, fill=color, font=font)

    return base
