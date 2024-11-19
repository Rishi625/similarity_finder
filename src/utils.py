import numpy as np
import json

def save_np_array_to_file(array, filename):
    np.save(filename, array)

def load_np_array_from_file(filename):
    return np.load(filename)

def save_json_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_json_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)