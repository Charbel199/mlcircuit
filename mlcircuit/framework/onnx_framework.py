import torch
import tensorflow as tf
import onnx
import logging
import os

logger = logging.getLogger(__name__)


def check_onnx_model(onnx_path: str) -> bool:
    try:
        if os.path.exists(onnx_path):
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            return True
        else:
            logger.error(f"Could not find ONNX model: {onnx_path}.")
            return False
    except  Exception as e:
        logger.error(f"Error while loading {onnx_path}: {str(e)}")
        return False




def to_onnx(model_type: str, input_shape, dtype, onnx_path: str, model_path: str = None, model = None):
    if 'torch' in model_type.lower():

        # Load model
        if model is None:
            model = torch.load(model_path)
            model.eval()

        # Input 
        input_tensor = torch.randn(*input_shape, dtype=dtype)

        # Conversion
        onnx_opset_version = 15
        torch.onnx.export(model,
                          input_tensor,
                          onnx_path,
                          export_params=True,
                          opset_version=onnx_opset_version,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}) # Batch size is variable


    elif 'tensorflow' in model_type.lower() or 'tf' in model_type.lower():
        # Load model
        if model is None:
            model = tf.keras.models.load_model(model_path)

        # Convert model to concrete function
        full_model = tf.function(lambda inputs: model(inputs))    
        full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

        # Persist input and output params
        input_names = [inp.name for inp in full_model.inputs]
        output_names = [out.name for out in full_model.outputs]
        print("Inputs:", input_names)
        print("Outputs:", output_names)

        # Freeze model
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        # Conversion 
        from tf2onnx import tf_loader
        from tf2onnx.tfonnx import process_tf_graph
        from tf2onnx.optimizer import optimize_graph
        from tf2onnx import utils, constants
        from tf2onnx.handler import tf_op
        extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')
        with tf_loader.tf_session(graph=tf_graph):
            g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)
        onnx_graph = optimize_graph(g)
        model_proto = onnx_graph.make_model("converted")

        utils.save_protobuf(onnx_path, model_proto)
    else:
        raise Exception(f"Model type not supported: {model_type}")
    
    onnx_model_health = check_onnx_model(onnx_path)
    if not onnx_model_health:
        raise Exception(f"Conversion to onnx was not successful")
    
    logger.info(f"Done converting from {model_path} to {onnx_path}")
    return onnx_path
