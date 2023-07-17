import pytest
import torch
import tensorflow as tf
from mlcircuit.framework.onnx_framework import to_onnx

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



def test_to_onnx_torch(torch_model):
    # Test converting a PyTorch model to ONNX
    input_shape = (1, 10)
    onnx_path = "test/data/torch_model.onnx"
    result = to_onnx(model_type = 'torch',
                    input_shape = input_shape,
                    dtype = torch.float32,
                    onnx_path = onnx_path,
                    model = torch_model)
    assert result == onnx_path


def test_to_onnx_torch_cnn(torch_cnn_model):
    # Test converting a PyTorch model to ONNX
    input_shape = (1, 3, 200, 200)
    onnx_path = "test/data/torch_model_cnn.onnx"
    result = to_onnx(model_type = 'torch',
                    input_shape = input_shape,
                    dtype = torch.float32,
                    onnx_path = onnx_path,
                    model = torch_cnn_model)
    assert result == onnx_path



def test_to_onnx_tensorflow(tensorflow_model):
    # Test converting a TensorFlow model to ONNX
    input_shape = (1, 10)
    onnx_path = "test/data/tensorflow_model.onnx"

    result = to_onnx(model_type = 'tf',
                      input_shape = input_shape,
                      dtype = tf.float32,
                      onnx_path = onnx_path,
                      model = tensorflow_model)
    
    assert result == onnx_path
