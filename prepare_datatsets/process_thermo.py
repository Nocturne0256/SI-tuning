from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from tqdm import tqdm
from pdb_to_pkl import pdb_to_pkl
import h5py


warnings.filterwarnings('ignore')

def convert_to_supported_dtype(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    elif isinstance(item, list):
        return np.array(item)
    elif isinstance(item, str):
        return np.string_(item)
    return item

def save_dict_to_hdf5(dic, file_path):
    with h5py.File(file_path, 'w') as hdf5_file:
        for key, value in dic.items():
            group = hdf5_file.create_group(key)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    sub_value = sub_value.numpy()  # 将 PyTorch Tensor 转换为 NumPy 数组
                if isinstance(sub_value, (list, tuple)):
                    sub_value = np.array(sub_value)  # 将列表或元组转换为 NumPy 数组
                group.create_dataset(sub_key, data=sub_value)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, f'{path}/{key}', item)
        else:
            item = convert_to_supported_dtype(item)
            h5file.create_dataset(f'{path}/{key}', data=item)

def load_dict_from_hdf5(filename):
    def recursively_load_dict_contents_from_group(h5file, path):
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.group.Group):
                # 递归处理嵌套字典
                ans[key] = recursively_load_dict_contents_from_group(h5file, f'{path}/{key}')
            else:
                # 加载非字典数据
                ans[key] = item[()]
                if isinstance(ans[key], bytes):
                    ans[key] = ans[key].decode('utf-8')
        return ans

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def process_key(key, data, data_saprot, pdb_dir):
    pdb_id = key
    chain_id = data[key]['chain']
    pdb_file = f"{pdb_dir}/{key}.pdb"

    targets = torch.tensor(data[key]['fitness'], dtype=torch.float32)
    saprot_sequence = data_saprot[key]['seq']

    protein_data = pdb_to_pkl(pdb_file, chain_id)

    if protein_data['sequence'] is None:
        print("error")
        return None

    protein_data['label'] = targets
    protein_data['sseq'] = saprot_sequence
    protein_data['plddt'] = data[key]['plddt']
    protein_data['key'] = key
    return key, protein_data

def process_thermo(cache_path, pdb_dir, npy_dir, saprot_dir):
    data_cache = []
    dic = {}

    data = np.load(npy_dir, allow_pickle=True).item()
    keys = [key for key in data.keys() if os.path.exists(os.path.join(pdb_dir, f"{key.split('_')[0]}.pdb"))]

    data_saprot = np.load(saprot_dir, allow_pickle=True).item()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_key, key, data, data_saprot, pdb_dir): key for key in keys}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                key, protein_data = result
                dic[key] = protein_data

    save_dict_to_hdf5(dic, cache_path)
    print(len(data_cache))

if __name__ == '__main__':
    process_thermo(cache_path='./datasets/Thermostability/test.h5', 
                   pdb_dir='./datasets/Thermostability/test',
                   npy_dir='./datasets/Thermostability/test.npy',
                   saprot_dir='./datasets/Thermostability/saprot.npy')
