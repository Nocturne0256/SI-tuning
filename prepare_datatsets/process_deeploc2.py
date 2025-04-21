from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from tqdm import tqdm
import h5py

import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pdb_to_pkl import *

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


def pdb_to_pkl_plddt(pdb_file, pkl_path=None, chain_id='A', parser=PDBParser()):
    structure = parser.get_structure('structure', pdb_file)
    
    # 处理大小写兼容
    if chain_id.islower():
        chain_id = chain_id.upper()
    if chain_id not in structure[0]:
        chain_id = chain_id.lower()
    if chain_id in structure[0]:
        chain_residues = list(structure[0][chain_id].get_residues())
    else:
        raise ValueError(f"Chain ID '{chain_id}' not found in the structure.")
    
    # 新建裁剪链（长度最多为 1024）
    chain = Chain.Chain("A")
    cnt = 0
    for residue in chain_residues:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            chain.add(residue)
            cnt += 1
            if cnt == 1024:
                break

    sequence = ''
    ca_coordinates = []
    side_chain_dihedrals = []
    backbone_dihedrals = []
    tau_angle = []
    theta_angle = []
    plddt = []

    for residue in chain:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            ca_atom = residue['CA']
            ca_coordinates.append(ca_atom.coord)
            side_chain_dihedrals.append(get_dihedral_angles(residue))
            plddt.append(ca_atom.get_bfactor())  # <-- 提取 plDDT

    # Backbone 二面角等
    polypeptides = PPBuilder().build_peptides(chain)
    for poly in polypeptides:
        phi_psi = poly.get_phi_psi_list()
        for phi, psi in phi_psi:
            if phi and psi:
                backbone_dihedrals.append((phi, psi))
            else:
                backbone_dihedrals.append((0, 0))

        tau = poly.get_tau_list()
        for angle in tau:
            tau_angle.append(angle if angle else 0)
        tau_angle += [0, 0, 0]

        theta = poly.get_theta_list()
        for angle in theta:
            theta_angle.append(angle if angle else 0)
        theta_angle += [0, 0]

        sequence += str(poly.get_sequence())

    len_seq = len(sequence)
    len_ca = len(ca_coordinates)
    len_tau = len(tau_angle)

    if len_tau < len_seq:
        tau_angle += [0] * (len_ca - len_tau)
        theta_angle += [0] * (len_ca - len_tau)
    elif len_tau > len_seq:
        tau_angle = tau_angle[:len_ca]
        theta_angle = theta_angle[:len_ca]

    if len_ca > len_seq:
        ca_coordinates = ca_coordinates[:len_seq]
        side_chain_dihedrals = side_chain_dihedrals[:len_seq]
        plddt = plddt[:len_seq]

    protein_data = {
        'sequence': sequence,
        'ca_coordinates': ca_coordinates,
        'side_chain_dihedrals': side_chain_dihedrals,
        'backbone_dihedrals': backbone_dihedrals,
        'tau_angle': tau_angle,
        'theta_angle': theta_angle,
        'plddt': plddt  
    }

    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(protein_data, f)

    return protein_data

LABEL_COLUMNS = [
    "Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
    "Mitochondrion", "Plastid", "Endoplasmic reticulum",
    "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
]

def process_key(key, row, pdb_dir):
    pdb_file = os.path.join(pdb_dir, f"{key}.pdb")
    if not os.path.exists(pdb_file):
        return None

    chain_id = 'A'

    protein_data = pdb_to_pkl_plddt(pdb_file, chain_id)
    if protein_data['sequence'] is None:
        return None

    # 提取多标签
    label = torch.tensor([row[col] for col in LABEL_COLUMNS], dtype=torch.float32)

    protein_data['label'] = label
    protein_data['sseq'] = row['Sequence']
    protein_data['fold'] = int(row['Partition'])
    protein_data['key'] = key

    return key, protein_data

def process_deeploc2(cache_path, pdb_dir, csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['ACC'].notna()]

    dic = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_key, row['ACC'], row, pdb_dir): row['ACC']
            for _, row in df.iterrows()
            if os.path.exists(os.path.join(pdb_dir, f"{row['ACC']}.pdb"))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                key, protein_data = result
                dic[key] = protein_data
            break
    save_dict_to_hdf5(dic, cache_path)
    print(f"✅ Saved {len(dic)} proteins to cache.")

if __name__ == '__main__':
    process_deeploc2(cache_path='./SI-tuning/cache/deeploc3.h5', 
                   pdb_dir='./SI-tuning/deeploc2/pdbs',
                   csv_path='./SI-tuning/deeploc2/Swissprot_Train_Validation_dataset.csv')