from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from tqdm import tqdm
from pdb_to_pkl import pdb_to_pkl
import h5py
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.MMCIFParser import MMCIFParser


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

import cProfile

pdb_dir = './datasets/GO/train_chain_fix'
pdb_files = set(os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb'))
cif_files = set(os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.cif'))


def preprocess_keys(keys):
    file_paths = {}
    for key in keys:
        pdb_id, chain_id = key.split('_')[0], key.split('_')[1].split('-')[0]
        chain_id = chain_id.upper()
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}_{chain_id}.pdb")
        if pdb_file not in pdb_files:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}_{chain_id.lower()}.pdb")
            if pdb_file not in pdb_files:
                if os.path.join(pdb_dir, f"{pdb_id}.cif") in cif_files:
                    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.cif")
                else:
                    print(pdb_file, 'not found')
        file_paths[key] = pdb_file
    return file_paths


import time

def process_keys(keys_chunk, file_paths, BP_data, MF_data, CC_data):
    parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)
    dic = {}
    for key in keys_chunk:
        try:
            # start_time = time.time()
            file_path = file_paths[key]
            chain_id = key.split('_')[1].split('-')[0]
            protein_data = pdb_to_pkl(file_path, chain_id=chain_id, parser=(cif_parser if file_path.endswith('.cif') else parser))
            protein_data['label_BP'] = torch.tensor(BP_data[key]['label'], dtype=torch.float32)
            protein_data['label_MF'] = torch.tensor(MF_data[key]['label'], dtype=torch.float32)
            protein_data['label_CC'] = torch.tensor(CC_data[key]['label'], dtype=torch.float32)
            protein_data['sseq'] = BP_data[key]['seq']
            # end_time = time.time()
            # print(end_time - start_time)
            dic[key] = protein_data
        except Exception as e:
            print(e)
    return dic



def process(cache_path, npy_dir):
    BP_data = np.load(os.path.join(npy_dir + '_BP.npy'), allow_pickle=True).item()
    CC_data = np.load(os.path.join(npy_dir + '_CC.npy'), allow_pickle=True).item()
    MF_data = np.load(os.path.join(npy_dir + '_MF.npy'), allow_pickle=True).item()
    keys = list(BP_data.keys())

    file_paths = preprocess_keys(keys)
    chunk_size = 20 # 调整这个值以找到最佳的性能
    keys_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    dic = {}
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_keys, keys_chunks, [file_paths]*len(keys_chunks),
                                         [BP_data]*len(keys_chunks), [MF_data]*len(keys_chunks), [CC_data]*len(keys_chunks)),
                            total=len(keys_chunks), desc="Processing PDB files"))
        # futures = [executor.submit(process_keys, keys_chunk, file_paths, BP_data, MF_data, CC_data) for keys_chunk in keys_chunks]
        # for future in tqdm(futures, desc="Processing PDB files"):
        #     dic.update(future.result())
    for result in results:
        dic.update(result)

    save_dict_to_hdf5(dic, cache_path)

if __name__ == '__main__':
    process(cache_path='./SI-tuning/cache/GO/train.h5', 
                   npy_dir='./datasets/GO/train')
    # process(cache_path='./SI-tuning/cache/GO/test.h5', 
    #                pdb_dir='./datasets/GO/test_chain_fix',
    #                npy_dir='./datasets/GO/test')
    # process(cache_path='./SI-tuning/cache/GO/valid.h5', 
    #                pdb_dir='./datasets/GO/valid_chain_fix',
    #                npy_dir='./datasets/GO/valid')

