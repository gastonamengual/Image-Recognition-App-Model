import io
import os

import numpy as np
import pytest

from object_detection_model.model import Model, ModelConfig


@pytest.fixture
def sample_img_url() -> str:
    current_dir = os.getcwd()
    return f"{current_dir}/tests/sample_images/computer.jpg"


def test_detect_objects(sample_img_url: str):
    with open(sample_img_url, "rb") as image_file:
        image_bytes = io.BytesIO(image_file.read()).read()
    processed_image = np.frombuffer(image_bytes, dtype=np.uint8)

    model = Model(
        ModelConfig(
            config_file_path="object_detection_model/model_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
            frozen_model_path="object_detection_model/model_files/frozen_inference_graph.pb",
            labels_path="object_detection_model/model_files/coco.names",
        )
    )
    model.detect_object(data=processed_image)
    assert type(image_bytes) == bytes
