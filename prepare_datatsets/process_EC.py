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

def process_key(key, pdb_dir, data, pdb_files, cif_files):
    parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)
    pdb_id, chain_id = key.split('_')[0], key.split('_')[1].split('-')[0]
    try:
        if chain_id.islower():
            chain_id = chain_id.upper()
        if f"{pdb_id}.pdb" in pdb_files:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            protein_data = pdb_to_pkl(pdb_file, chain_id=chain_id, parser=parser)
        elif f"{pdb_id}.pdb" in pdb_files:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            protein_data = pdb_to_pkl(pdb_file, chain_id=chain_id, parser=parser)
        elif f"{pdb_id}.cif" in cif_files:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.cif")
            protein_data = pdb_to_pkl(pdb_file, chain_id=chain_id, parser=cif_parser)
        else:
            print("error key: ", key)
            return None, None
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return f"Error processing {pdb_file}: {e}"

    targets = torch.tensor(data[key]['label'], dtype=torch.float32)
    protein_data['label'] = targets
    protein_data['sseq'] = data[key]['seq']
    return key, protein_data

def process_EC(cache_path, pdb_dir, npy_dir):
    data = np.load(npy_dir, allow_pickle=True).item()
    keys = list(data.keys())

    # 缓存目录中的文件列表
    dic = {}
    pdb_files = set(f for f in os.listdir(pdb_dir) if f.endswith('.pdb'))
    cif_files = set(f for f in os.listdir(pdb_dir) if f.endswith('.cif'))

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_key, keys, [pdb_dir]*len(keys), [data]*len(keys), 
                                         [pdb_files]*len(keys), [cif_files]*len(keys)), 
                            total=len(keys), desc="Processing PDB files"))

    for key, protein_data in results:
        if key is not None:
            dic[key] = protein_data

    save_dict_to_hdf5(dic, cache_path)

if __name__ == '__main__':
    process_EC(cache_path='./SI-tuning/cache/EC/train.h5', 
                   pdb_dir='./datasets/EC/train',
                   npy_dir='./datasets/EC/train.npy')
    process_EC(cache_path='./SI-tuning/cache/EC/test.h5', 
                   pdb_dir='./datasets/EC/test',
                   npy_dir='./datasets/EC/test.npy')
    process_EC(cache_path='./SI-tuning/cache/EC/valid.h5', 
                   pdb_dir='./datasets/EC/valid',
                   npy_dir='./datasets/EC/valid.npy')

