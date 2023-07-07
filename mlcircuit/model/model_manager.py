import os
import shutil
LOCAL_ONNX_FOLDER = os.path.join(os.path.dirname(os.getcwd()),'data/models')
def get_local_onnx_files():
    folder_path = LOCAL_ONNX_FOLDER
    file_list = os.listdir(folder_path)
    onnx_files = [file for file in file_list if file.endswith('.onnx')]
    return onnx_files

def add_local_onnx_file(onnx_path):
    file_name = os.path.basename(onnx_path)
    destination_path = os.path.join(LOCAL_ONNX_FOLDER, file_name)
    shutil.copyfile(onnx_path, destination_path)
    print(f"ONNX file '{file_name}' added to '{LOCAL_ONNX_FOLDER}'.")

if __name__ == "__main__":
    print(get_local_onnx_files())
    add_local_onnx_file('/home/charbel199/projs/mlcircuit/test/data/torch_model.onnx')
    print(get_local_onnx_files())