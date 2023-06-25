import torch
import tensorflow as tf
import onnx
import logging

from enum import Enum

logger = logging.getLogger(__name__)


class ModelTypes(Enum):
    TORCH = 1
    TENSORFLOW = 2


class Framework:
    def __init__(self, model):
        self.model = model
        self.packaged_model = None
        self.model_type = None
        self.input_size = None
        self.onnx_path = None
        self.dtype = None
        self._check_model_info(self.model)

    def _check_model_info(self, model):
        if isinstance(model, torch.nn.Module):
            self.model_type = ModelTypes.TORCH
            self.input_size = next(model.parameters()).shape[1:]
            self.dtype = next(model.parameters()).dtype
        elif isinstance(model, tf.keras.Model):
            self.model_type = ModelTypes.TENSORFLOW
            self.input_size = model.input_shape[1:]
            self.dtype = model.weights[0].dtype
        else:
            logger.error(f"Mode type: {type(model)} not supported.")
            raise Exception(f"Mode type: {type(model)} not supported.")

    def _check_onnx_model(self):
        if self.onnx_path:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model)
        else:
            logger.error(f"ONNX model not generated yet.")

    def to_onnx(self, onnx_path: str):
        if self.model_type == ModelTypes.TORCH:
            # TODO: Check onnx2torch
            torch.onnx.export(self.model,
                              torch.randn(1, *self.input_size, dtype=self.dtype),
                              onnx_path,
                              export_params=True)
        elif self.model == ModelTypes.TENSORFLOW:
            import tf2onnx
            input_signature = (tf.TensorSpec((None, *self.input_size), self.dtype, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature)
            onnx.save(onnx_model, onnx_path)
        self.onnx_path = onnx_path
        return onnx_path
