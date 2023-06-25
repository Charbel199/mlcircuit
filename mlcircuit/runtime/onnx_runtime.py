import os
import onnxruntime as ort
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class OnnxRunner:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self._set_providers()
        self.inference_session = None

    def _set_providers(self):
        if os.getenv("TENSORRT") == "1":
            self.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        elif os.getenv("GPU") == "1":
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

    def start_inference_session(self):
        self.inference_session = ort.InferenceSession(self.onnx_path, providers=self.providers)

    def stop_inference_session(self):
        del self.inference_session
        self.inference_session = None

    def __call__(self, x):
        if self.inference_session is None:
            raise Exception('Inference sessions has not been initiated')
        outputs = self.inference_session.run(None, {'input': x})
        return outputs
