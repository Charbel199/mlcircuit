import pytest
import torch
import tensorflow as tf
from mlcircuit.framework.onnx_framework import Framework, ModelTypes

import torch.nn as nn


class SmallTorchCNN(nn.Module):
    def __init__(self):
        super(SmallTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 50 * 50, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def torch_model():
    # Create a dummy PyTorch model for testing
    model = torch.nn.Linear(10, 5)
    return model


@pytest.fixture
def torch_cnn_model():
    # Create a dummy PyTorch model for testing
    model = SmallTorchCNN()
    return model


@pytest.fixture
def tensorflow_model():
    # Create a dummy TensorFlow model for testing
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5, input_shape=(10,)))
    return model


def test_model_info_torch(torch_model):
    # Test checking model info for a PyTorch model
    input_shape = (10,)
    framework = Framework(torch_model, input_shape)
    assert framework.model_type == ModelTypes.TORCH
    assert framework.input_shape == (10,)
    assert framework.dtype == torch.float32


def test_cnn_model_info_torch(torch_cnn_model):
    # Test checking model info for a PyTorch model
    input_shape = (3, 200, 200)
    assert torch_cnn_model(torch.randn(1, *input_shape)) is not None
    framework = Framework(torch_cnn_model, input_shape)
    assert framework.model_type == ModelTypes.TORCH
    assert framework.input_shape == (3, 200, 200)
    assert framework.dtype == torch.float32


def test_model_info_tensorflow(tensorflow_model):
    # Test checking model info for a TensorFlow model
    input_shape = (10,)
    framework = Framework(tensorflow_model, input_shape)
    assert framework.model_type == ModelTypes.TENSORFLOW
    assert framework.input_shape == (10,)
    assert framework.dtype == tf.float32


def test_model_info_invalid():
    # Test checking model info for an unsupported model type
    model = "Invalid model type"
    input_shape = (10,)
    with pytest.raises(Exception):
        Framework(model, input_shape)


def test_to_onnx_torch(torch_model):
    # Test converting a PyTorch model to ONNX
    input_shape = (10,)
    framework = Framework(torch_model, input_shape)
    onnx_path = "test/data/torch_model.onnx"
    result = framework.to_onnx(onnx_path)
    assert result == onnx_path
    framework.check_onnx_model()


def test_to_onnx_torch_cnn(torch_cnn_model):
    # Test converting a PyTorch model to ONNX
    input_shape = (3, 200, 200)
    framework = Framework(torch_cnn_model, input_shape)
    onnx_path = "test/data/torch_model_cnn.onnx"
    result = framework.to_onnx(onnx_path)
    assert result == onnx_path
    framework.check_onnx_model()


def test_to_onnx_tensorflow(tensorflow_model):
    # Test converting a TensorFlow model to ONNX
    input_shape = (10,)
    framework = Framework(tensorflow_model, input_shape)
    onnx_path = "test/data/tensorflow_model.onnx"
    result = framework.to_onnx(onnx_path)
    assert result == onnx_path
    framework.check_onnx_model()
