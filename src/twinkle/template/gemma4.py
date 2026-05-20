import json
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union
from dataclasses import dataclass, field, fields
from PIL import Image
from copy import deepcopy

from twinkle import remote_class, requires
from twinkle.template import Template
from twinkle.data_format import InputFeature, Message, Trajectory

@remote_class()
class Gemma4Template(Template):
    """Processor for Google Gemma4 series."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use original Template

    def _build_standard_messages(self, trajectory: Trajectory) -> List[Trajectory]:
        # Extract trajectory-level media
        extracted_images = trajectory.pop('images', None) or [
            img for msg in trajectory['messages']
            for img in msg.get('images', []) or []
        ]
        extracted_videos = trajectory.pop('videos', None) or [
            video for msg in trajectory['messages']
            for video in msg.get('videos', []) or []
        ]
        extracted_audios = trajectory.pop('audios', None) or [
            audio for msg in trajectory['messages']
            for audio in msg.get('audios', []) or []
        ]
        images = self.preprocess_images(extracted_images)
        videos = self.preprocess_videos(extracted_videos)
        audios = self.preprocess_audios(extracted_audios)

        trajectory['messages'] = self._process_mm_messages(trajectory['messages'], images, videos, audios)
        if not self.is_mm:
            for message in trajectory['messages']:
                message['content'] = message['content'][0]['text']
        return [trajectory]