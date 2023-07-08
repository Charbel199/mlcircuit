import os
from mlcircuit.utils import get_data_path
import shutil
LOCAL_ONNX_FOLDER = os.path.join(get_data_path(), 'models')

def get_local_onnx_files():
    file_list = os.listdir(LOCAL_ONNX_FOLDER)
    onnx_files = [file for file in file_list if file.endswith('.onnx')]
    return onnx_files

def add_local_onnx_file(onnx_path):
    file_name = os.path.basename(onnx_path)
    destination_path = os.path.join(LOCAL_ONNX_FOLDER, file_name)
    shutil.copyfile(onnx_path, destination_path)
    print(f"ONNX file '{file_name}' added to '{LOCAL_ONNX_FOLDER}'.")

def remove_local_onnx_file(onnx_name):
    onnx_path = os.path.join(LOCAL_ONNX_FOLDER, onnx_name)
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"ONNX file '{onnx_path}' remove from '{LOCAL_ONNX_FOLDER}'.")
    else:
        print(f"ONNX file '{onnx_path}' not found in '{LOCAL_ONNX_FOLDER}'.")

if __name__ == "__main__":
    print(get_local_onnx_files())
    add_local_onnx_file('/home/charbel199/projs/mlcircuit/test/data/torch_model.onnx')
    print(get_local_onnx_files())
    remove_local_onnx_file('torch_model.onnx')
    print(get_local_onnx_files())