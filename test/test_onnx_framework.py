import pytest
import torch
import tensorflow as tf
from mlcircuit.framework.onnx_framework import Framework, ModelTypes


@pytest.fixture
def torch_model():
    # Create a dummy PyTorch model for testing
    model = torch.nn.Linear(10, 5)
    return model


@pytest.fixture
def tensorflow_model():
    # Create a dummy TensorFlow model for testing
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5, input_shape=(10,)))
    return model


def test_model_info_torch(torch_model):
    # Test checking model info for a PyTorch model
    framework = Framework(torch_model)
    assert framework.model_type == ModelTypes.TORCH
    assert framework.input_size == (10,)
    assert framework.dtype == torch.float32


def test_model_info_tensorflow(tensorflow_model):
    # Test checking model info for a TensorFlow model
    framework = Framework(tensorflow_model)
    assert framework.model_type == ModelTypes.TENSORFLOW
    assert framework.input_size == (10,)
    assert framework.dtype == tf.float32


def test_model_info_invalid():
    # Test checking model info for an unsupported model type
    model = "Invalid model type"
    with pytest.raises(Exception):
        Framework(model)


def test_to_onnx_torch(torch_model):
    # Test converting a PyTorch model to ONNX
    framework = Framework(torch_model)
    onnx_path = "torch_model.onnx"
    result = framework.to_onnx(onnx_path)
    assert result == onnx_path
    # Additional assertions or checks for the generated ONNX model


def test_to_onnx_tensorflow(tensorflow_model):
    # Test converting a TensorFlow model to ONNX
    framework = Framework(tensorflow_model)
    onnx_path = "tensorflow_model.onnx"
    result = framework.to_onnx(onnx_path)
    assert result == onnx_path
    # Additional assertions or checks for the generated ONNX model
