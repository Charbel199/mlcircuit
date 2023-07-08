import os

def get_data_path():
    package_dir = os.path.dirname(__file__)  
    data_dir = os.path.join(package_dir, 'data')
    return data_dir