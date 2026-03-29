from __future__ import annotations

import copy
from io import BytesIO

from recipe.gui_project.util.image_process import draw_bbox_on_image
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.vision_utils import process_image


def draw_bbox_by_tag(
    *,
    current_epoch: int,
    current_step: int,
    total_epochs: int,
    total_steps: int,
    draw_tag: str | None,
    gt_bbox_xyxy: list[float] | tuple[float, float, float, float],
    image,
):
    normalized_tag = (draw_tag or "none").strip().lower()
    if normalized_tag in {"", "none", "off"}:
        return image
    if normalized_tag == "first_two_epoch":
        return _draw_bbox_first_two_epoch(
            current_epoch=current_epoch,
            current_step=current_step,
            total_epochs=total_epochs,
            total_steps=total_steps,
            gt_bbox_xyxy=gt_bbox_xyxy,
            image=image,
        )
    raise ValueError(f"Unsupported gt bbox draw tag: {draw_tag}")


def _draw_bbox_first_two_epoch(
    *,
    current_epoch: int,
    current_step: int,
    total_epochs: int,
    total_steps: int,
    gt_bbox_xyxy: list[float] | tuple[float, float, float, float],
    image,
):
    del current_step, total_epochs, total_steps

    if current_epoch == 1:
        return draw_bbox_on_image(image=image, bbox_xyxy=gt_bbox_xyxy, transparency=0.0)
    if current_epoch == 2:
        return draw_bbox_on_image(image=image, bbox_xyxy=gt_bbox_xyxy, transparency=0.5)
    return image


class GuiGroundingBboxDrawDataset(RLHFDataset):
    """RL dataset that optionally draws gt bbox overlays on GUI grounding images during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_bbox_draw_tag = (self.config.get("gt_bbox_draw_tag", "none") or "none").strip().lower()
        self.runtime_progress = {
            "split": "unknown",
            "epoch": 1,
            "global_step": 1,
            "step_in_epoch": 1,
            "total_epochs": 1,
            "total_steps": 1,
        }

    def set_runtime_progress(
        self,
        *,
        split: str | None = None,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        total_epochs: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        if split is not None:
            self.runtime_progress["split"] = str(split)
        self.runtime_progress["epoch"] = max(1, int(epoch))
        self.runtime_progress["global_step"] = max(1, int(global_step))
        self.runtime_progress["step_in_epoch"] = max(1, int(step_in_epoch))
        if total_epochs is not None:
            self.runtime_progress["total_epochs"] = max(1, int(total_epochs))
        if total_steps is not None:
            self.runtime_progress["total_steps"] = max(1, int(total_steps))

    def _should_draw_bbox(self, row_dict: dict) -> bool:
        if self.runtime_progress["split"] != "train":
            return False
        if self.gt_bbox_draw_tag in {"", "none", "off"}:
            return False

        extra_info = row_dict.get("extra_info") or {}
        split = str(extra_info.get("split", "")).lower()
        if split and split != "train":
            return False

        return bool(row_dict.get(self.image_key)) and extra_info.get("gt_bbox_pixels") is not None

    def _build_bbox_drawn_images(self, row_dict: dict, *, current_epoch: int, current_step: int) -> list[dict]:
        extra_info = row_dict.get("extra_info") or {}
        gt_bbox_xyxy = extra_info["gt_bbox_pixels"]
        drawn_images = []

        for image_item in row_dict[self.image_key]:
            image_item_copy = copy.deepcopy(image_item)
            image_item_copy.pop("image", None)
            pil_image = process_image(image_item_copy, image_patch_size=self.image_patch_size)
            processed_image = draw_bbox_by_tag(
                current_epoch=current_epoch,
                current_step=current_step,
                total_epochs=self.runtime_progress["total_epochs"],
                total_steps=self.runtime_progress["total_steps"],
                draw_tag=self.gt_bbox_draw_tag,
                gt_bbox_xyxy=gt_bbox_xyxy,
                image=pil_image,
            )

            image_buffer = BytesIO()
            processed_image.save(image_buffer, format="PNG")

            new_image_item = {key: value for key, value in image_item.items() if key != "image"}
            new_image_item["bytes"] = image_buffer.getvalue()
            drawn_images.append(new_image_item)

        return drawn_images

    def preprocess_row_dict(self, row_dict: dict) -> dict:
        if self.runtime_progress["split"] != "train":
            return row_dict

        current_epoch = self.runtime_progress["epoch"]
        current_step = self.runtime_progress["global_step"]

        if self._should_draw_bbox(row_dict):
            row_dict = copy.deepcopy(row_dict)
            row_dict[self.image_key] = self._build_bbox_drawn_images(
                row_dict=row_dict,
                current_epoch=current_epoch,
                current_step=current_step,
            )
        return row_dict
