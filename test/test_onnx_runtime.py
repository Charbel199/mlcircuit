import pytest
import torch
from mlcircuit.framework.onnx_framework import Framework
from mlcircuit.runtime.onnx_runtime import OnnxRunner
import onnxruntime as ort
import os

OUTPUT_SIZE = 5


@pytest.fixture(scope="session", autouse=True)
def to_onnx_torch():
    input_shape = (10,)
    model = torch.nn.Linear(10, OUTPUT_SIZE)
    framework = Framework(model, input_shape)
    onnx_path = "test/data/torch_model_runtime.onnx"
    framework.to_onnx(onnx_path)


@pytest.fixture
def onnx_path():
    return "test/data/torch_model_runtime.onnx"


def test_onnxrunner_start_stop_inference_session(onnx_path):
    runner = OnnxRunner(onnx_path)
    runner.start_inference_session()
    assert isinstance(runner.inference_session, ort.InferenceSession)
    runner.stop_inference_session()
    assert runner.inference_session is None


def test_onnxrunner_call(onnx_path):
    runner = OnnxRunner(onnx_path)
    runner.start_inference_session()
    input_shape = (1, 10)
    input_tensor = torch.rand(*input_shape).numpy()
    outputs = runner(input_tensor)
    assert len(outputs[0][0]) == OUTPUT_SIZE
