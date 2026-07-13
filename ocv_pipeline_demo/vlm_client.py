"""Qwen3-VL scene understanding via vLLM OpenAI-compatible API."""

import base64
import io
import time

import cv2
import numpy as np
from openai import OpenAI

import config


class VLMClient:
    def __init__(self, base_url=None, model_name=None):
        self.base_url = base_url or config.VLLM_BASE_URL
        self.model_name = model_name or config.VLLM_MODEL_NAME
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

    def describe_roi(self, roi_image, prompt=None):
        """Send a cropped ROI image to Qwen3-VL and get a description.

        Args:
            roi_image: BGR numpy array of the cropped region
            prompt: optional custom prompt

        Returns:
            description string
        """
        _, buf = cv2.imencode(".jpg", roi_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt or config.VLM_PROMPT,
                            },
                        ],
                    }
                ],
                max_tokens=config.VLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[VLM error: {e}]"

    def describe_rois(self, frame, detections, top_k=None):
        """Describe the top-K detected regions.

        Args:
            frame: original BGR frame
            detections: list of (x1, y1, x2, y2, score, class_id)
            top_k: max number of ROIs to describe

        Returns:
            list of (detection, description) tuples
        """
        top_k = top_k or config.VLM_TOP_K_ROIS
        sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)[:top_k]

        results = []
        for det in sorted_dets:
            x1, y1, x2, y2 = [int(v) for v in det[:4]]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            desc = self.describe_roi(roi)
            results.append((det, desc))

        return results

    def health_check(self):
        """Check if the vLLM server is ready."""
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
