import glob
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm
from torch.utils.data import DataLoader
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
                    pass
        return ans

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

class ProteinDataset_DeepLoc(Dataset):
    def __init__(self, pdb_dir='', npy_dir='', c=2,  cls2_npy_dir= None, cache_path=None, part = False):
        self.data_cache = []
        if os.path.exists(cache_path):
            dic = load_dict_from_hdf5(cache_path)
            for k,v in dic.items():
                self.data_cache.append(v)
            return
        self.pdb_dir = pdb_dir
        #self.pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        self.parser = PDBParser(QUIET=True)
        self.npy_dir = npy_dir
        self.data = np.load(npy_dir, allow_pickle = True).item()
        if cls2_npy_dir:
            self.data_cls2 = np.load(cls2_npy_dir, allow_pickle = True).item()
        else:
            self.data_cls2 = self.data
        self.keys = list(self.data.keys())
        self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb"))]
        self.data_cache = []
        self.cls = c
        

        dic = {}
        for key in tqdm(self.keys):
            if self.cls == 2:
                chain_id = self.data[key]['chain']
            elif self.cls == 10 and key in self.data_cls2:
                chain_id = self.data_cls2[key]['chain']
            else:
                chain_id = None
            
            pdb_file = f"{self.pdb_dir}/{key}.pdb"
            targets = torch.tensor(self.data[key]['label'], dtype=torch.float32)
            plddt = torch.tensor(self.data[key]['plddt'], dtype=torch.float32)
            protein_data = pdb_to_pkl(pdb_file, chain_id)
            protein_data['label'] = targets
            protein_data['key'] = key
            protein_data['plddt'] = plddt
            protein_data['sseq'] = self.data[key]['seq']
            if protein_data['sequence'] is None:
                continue
            dic[key] = protein_data
        
        save_dict_to_hdf5(dic, cache_path)
        
        for k,v in dic.items():
            self.data_cache.append(v)
        print(len(self.data_cache))
        # from IPython import embed; embed(); assert 0


    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        # 解析pdb_id和chain_id
        return self.data_cache[idx]


class ProteinDataset_DeepLoc_cls10(Dataset):
    def __init__(self, pdb_dir='', npy_dir='', c=10,  cls2_npy_dir= None, cache_path=None):
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self.data = np.load(npy_dir, allow_pickle = True).item()
        self.keys = list(self.data.keys())
        self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")) or os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif"))]
        self.cache_path = cache_path
        self.data_cls2 = np.load(cls2_npy_dir, allow_pickle = True).item()
        self.cls = c

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        cache_key = f"{key}.npy"
        cache_path = os.path.join(self.cache_path, cache_key)
        if os.path.exists(cache_path):
            try:
                protein_data = np.load(cache_path, allow_pickle=True).item()
                return protein_data
            except:
                return None
            

        if self.cls == 10 and key in self.data_cls2:
            chain_id = self.data_cls2[key]['chain']
        else:
            chain_id = 'A'

        pdb_file = f"{self.pdb_dir}/{key}.pdb"
        targets = torch.tensor(self.data[key]['label'], dtype=torch.float32)
        plddt = torch.tensor(self.data[key]['plddt'], dtype=torch.float32)
        protein_data = pdb_to_pkl(pdb_file, chain_id = chain_id)
        protein_data['label'] = targets
        protein_data['key'] = key
        protein_data['plddt'] = plddt
        protein_data['sseq'] = self.data[key]['seq']
        if protein_data['sequence'] is None:
            return None
        # 解析pdb_id和chain_id
        np.save(cache_path, protein_data)
        return protein_data

class ProteinDataset_EC(Dataset):
    def __init__(self, cache_path, pdb_dir=None, npy_dir=None, part=False):
        self.cache_path = cache_path
        self.data_cache = []
        if os.path.exists(self.cache_path):
            dic = load_dict_from_hdf5(self.cache_path)
            for k,v in dic.items():
                self.data_cache.append(v)
            return
        else:
            print("Cache not exist!")
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self.data = np.load(npy_dir, allow_pickle = True).item()
        self.keys = list(self.data.keys())
        self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")) or os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif"))]

    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        return self.data_cache[idx]

class ProteinDataset_GO(Dataset):
    def __init__(self, pdb_dir, npy_dir, cache_path, task = 'BP', part=False):
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        npy_dir = cache_path
        self.data = np.load(npy_dir+'_BP.npy', allow_pickle = True).item()
        self.CC_data = np.load(npy_dir+'_CC.npy', allow_pickle = True).item()
        self.MF_data = np.load(npy_dir+'_MF.npy', allow_pickle = True).item()
        self.keys = list(self.data.keys())
        #self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")) or os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif"))]
        self.cache_path = cache_path
        self.task = task

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        cache_key = f"{key}.npy"
        cache_path = os.path.join(self.cache_path, cache_key)
        if os.path.exists(cache_path):
            try:
                protein_data = np.load(cache_path, allow_pickle=True).item()
            except:
                return None
        else:
            pdb_id, chain_id = key.split('_')[0], key.split('_')[1].split('-')[0]
            if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")):
                parser = self.parser
                pdb_file = f"{self.pdb_dir}/{pdb_id}.pdb"
            elif os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif")):
                parser = self.cif_parser
                pdb_file = f"{self.pdb_dir}/{pdb_id}.cif"
            protein_data = pdb_to_pkl(pdb_file, chain_id=chain_id, parser=parser)
            BP_targets = torch.tensor(self.data[key]['label'], dtype=torch.float32)
            MF_targets = torch.tensor(self.MF_data[key]['label'], dtype=torch.float32)
            CC_targets = torch.tensor(self.CC_data[key]['label'], dtype=torch.float32)
            protein_data['label_BP'] = BP_targets
            protein_data['label_MF'] = MF_targets
            protein_data['label_CC'] = CC_targets
            protein_data['sseq'] = self.data[key]['seq']
            np.save(cache_path, protein_data)

        protein_data['label'] = protein_data['label_' + self.task]
        return protein_data

class ProteinDataset_MetalIonBinding(Dataset):
    def __init__(self, cache_path, pdb_dir=None, npy_dir=None, re_cache=False ):
        self.data_cache = []
        dic = {}
        self.cache_path = cache_path
        if os.path.exists(self.cache_path) and not re_cache:
            dic = load_dict_from_hdf5(self.cache_path)
            for k,v in dic.items():
                if len(v['backbone_dihedrals']) != 0:
                    self.data_cache.append(v)
            return
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self.npy_dir = npy_dir
        self.data = np.load(npy_dir, allow_pickle = True).item()
        self.keys = list(self.data.keys())
            
        self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")) or os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif"))]
        

        
        for key in tqdm(self.keys):
            chain_id = key.split('_')[1]
            chain_id = chain_id.split('-')[0]
            pdb_id = key.split('_')[0]
            if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb")):
                parser = self.parser
                pdb_file = f"{self.pdb_dir}/{pdb_id}.pdb"
                protein_data = pdb_to_pkl(pdb_file, chain_id = chain_id, parser=parser)
            elif os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.cif")):
                parser = self.cif_parser
                pdb_file = f"{self.pdb_dir}/{pdb_id}.cif"
                protein_data = pdb_to_pkl(pdb_file, chain_id = chain_id, parser=parser)
            else:
                print("Error!")
                return

            targets = torch.tensor(self.data[key]['label'], dtype=torch.float32)
            # saprot_sequence = data_saprot[key]['seq']

            protein_data['label'] = targets
            protein_data['sseq'] = self.data[key]['seq']
            # protein_data['saprot_sequence'] = saprot_sequence

            if protein_data['sequence'] is None:
                continue
            dic[key] = protein_data
        
        save_dict_to_hdf5(dic, self.cache_path)
        for k,v in dic.items():
            self.data_cache.append(v)
        print(len(self.data_cache))

    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        return self.data_cache[idx]
    
class ProteinDataset_Thermostability(Dataset):
    def __init__(self, cache_path, pdb_dir=None, npy_dir=None, saprot_dir=None):
        self.cache_path = cache_path
        self.data_cache = []
        dic = {}
        if os.path.exists(self.cache_path):
            dic = load_dict_from_hdf5(self.cache_path)
            for k,v in dic.items():
                self.data_cache.append(v)
            return
        
        self.pdb_dir = pdb_dir
        #self.pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        self.parser = PDBParser(QUIET=True)
        self.npy_dir = npy_dir
        self.data = np.load(npy_dir, allow_pickle = True).item()
        self.keys = list(self.data.keys())
        self.keys = [key for key in self.keys if os.path.exists(os.path.join(self.pdb_dir, f"{key.split('_')[0]}.pdb"))]
    
        data_saprot = np.load(saprot_dir, allow_pickle = True).item()
        
        for key in tqdm(self.keys):
            pdb_id = key
            chain_id = self.data[key]['chain']
            pdb_file = f"{self.pdb_dir}/{key}.pdb"

            targets = torch.tensor(self.data[key]['fitness'], dtype=torch.float32)
            saprot_sequence = data_saprot[key]['seq']

            try:
                protein_data = pdb_to_pkl(pdb_file, chain_id)
            except:
                from IPython import embed; embed(); assert 0
            protein_data['label'] = targets
            protein_data['saprot_sequence'] = saprot_sequence

            if protein_data['sequence'] is None:
                continue
            dic[key] = protein_data
        
        save_dict_to_hdf5(dic, self.cache_path)
        
        for k,v in dic.items():
            self.data_cache.append(v)
        print(len(self.data_cache))

    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        # 解析pdb_id和chain_id
        return self.data_cache[idx]

class ProteinDataset_DeepLoc_2(Dataset):
    def __init__(self, cache_path, fold=0, train=True):
        self.cache_path = cache_path
        self.fold = fold
        self.train = train
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)
        if train:
            self.data = {k: v for k, v in self.data.items() if v['fold'] != fold}
        else:
            self.data = {k: v for k, v in self.data.items() if v['fold'] == fold}
        # Convert dictionary to list
        self.data = list(self.data.values())
        # 将one-hot标签转换为数字标签
        for item in self.data:
            if isinstance(item['label'], torch.Tensor):
                item['label'] = torch.argmax(item['label']).item()
            elif isinstance(item['label'], (list, np.ndarray)):
                item['label'] = np.argmax(item['label'])
        pass
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       return self.data[idx]

def collate_protein_data_plddt(batch):
    if None in batch:
        print('dirty data')
        return None
    # batch是一个列表，其中每个元素都是__getitem__返回的一个样本
    cutoff = 1022
    for i in range(len(batch)):
        if len(batch[i]['sequence'])>cutoff:
            batch[i]['sequence'] = batch[i]['sequence'][:cutoff]
            #batch[i]['sseq'] = batch[i].get('sseq', batch[i].get('saprot_sequence', []))[:cutoff*2]
            batch[i]['ca_coordinates'] = batch[i]['ca_coordinates'][:cutoff]
            batch[i]['side_chain_dihedrals'] = batch[i]['side_chain_dihedrals'][:cutoff]
            batch[i]['backbone_dihedrals'] = batch[i]['backbone_dihedrals'][:cutoff]
            batch[i]['tau_angle'] = batch[i]['tau_angle'][:cutoff]
            batch[i]['theta_angle'] = batch[i]['theta_angle'][:cutoff]
            batch[i]['plddt'] = batch[i]['plddt'][:cutoff]
    
    sequences = [item['sequence'] for item in batch]
    #ssequences = [item['sseq'] for item in batch]
    ca_coordinates = [item['ca_coordinates'] for item in batch]
    side_chain_dihedrals = [item['side_chain_dihedrals'] for item in batch]
    backbone_dihedrals = [item['backbone_dihedrals'] for item in batch]
    tau_angles = [item['tau_angle'] for item in batch]
    theta_angles = [item['theta_angle'] for item in batch]
    plddts = [item['plddt'] for item in batch]
    labels = [item['label'] for item in batch]

    # 找出序列中最长的长度，用于后续的padding
    max_length = max(len(seq) for seq in sequences)
    padded_ca_coordinates = []
    padded_side_chain_dihedrals = []
    padded_backbone_dihedrals = []
    padded_tau_angles = []
    padded_theta_angles = []
    padded_plddt = []
    #padded_sseq = []
    padded_sequences = []
    padded_labels = []
    
    for label, seq, ca, sc_dih, bb_dih, tau, theta, plddt in zip(labels, sequences, ca_coordinates, 
                                                   side_chain_dihedrals, backbone_dihedrals, tau_angles, theta_angles, plddts):
        ca = torch.tensor(ca, dtype=torch.float32)
        sc_dih = torch.tensor(sc_dih, dtype=torch.float32)
        bb_dih = torch.tensor(bb_dih, dtype=torch.float32)
        tau = torch.tensor(tau, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)
        plddt = torch.tensor(plddt, dtype=torch.float32)

        if len(seq) == 0:
            return None
            continue
        if len(ca) > max_length:
            ca = ca[:max_length]
        if len(tau) > max_length:
            tau = tau[:max_length]
        if len(sc_dih) > max_length:
            sc_dih = sc_dih[:max_length]
        if len(plddt) > max_length:
            plddt = plddt[:max_length]
        
        padding_ca = torch.cat([ca, torch.zeros((max_length - len(ca), 3))])
        padding_sc_dih = torch.cat([sc_dih, torch.zeros((max_length - len(sc_dih), 4))])
        padding_bb_dih = torch.cat([bb_dih, torch.zeros((max_length - len(bb_dih), 2))])
        padding_tau = torch.cat([tau, torch.zeros(max_length - len(tau))])
        padding_theta = torch.cat([theta, torch.zeros(max_length - len(theta))])
        padding_plddt = torch.cat([plddt, torch.zeros(max_length - len(plddt))])
        
        padded_sequences.append(seq)
        #padded_sseq.append(sseq)
        padded_ca_coordinates.append(padding_ca)
        padded_side_chain_dihedrals.append(padding_sc_dih)
        padded_backbone_dihedrals.append(padding_bb_dih)
        padded_tau_angles.append(padding_tau)
        padded_theta_angles.append(padding_theta)
        padded_labels.append(label)
        padded_plddt.append(padding_plddt)

    # 转换为Tensor，并使用stack进行堆叠，以形成批量
    padded_ca_coordinates = torch.stack(padded_ca_coordinates)
    padded_side_chain_dihedrals = torch.stack(padded_side_chain_dihedrals)
    padded_backbone_dihedrals = torch.stack(padded_backbone_dihedrals)
    padded_tau_angles = torch.stack(padded_tau_angles)
    padded_theta_angles = torch.stack(padded_theta_angles)
    padded_plddt = torch.stack(padded_plddt)

    if str(type(padded_labels[0])) == "<class 'torch.Tensor'>":
        labels = torch.stack(padded_labels)
    else:
        labels = torch.tensor(padded_labels)
    
    return {
        'sequence': padded_sequences,
        'ca_coordinates': padded_ca_coordinates,
        'side_chain_dihedrals': padded_side_chain_dihedrals,
        'backbone_dihedrals': padded_backbone_dihedrals,
        'tau_angle': padded_tau_angles,
        'theta_angle': padded_theta_angles,
        'label': labels,
        'plddt' : padded_plddt
    }


def collate_protein_data_sseq(batch):
    if None in batch:
        print('dirty data')
        return None
    # batch是一个列表，其中每个元素都是__getitem__返回的一个样本
    cutoff = 1022
    for i in range(len(batch)):
        if len(batch[i]['sequence'])>cutoff:
            batch[i]['sequence'] = batch[i]['sequence'][:cutoff]
            batch[i]['sseq'] = batch[i].get('sseq', batch[i].get('saprot_sequence', []))[:cutoff*2]
            batch[i]['ca_coordinates'] = batch[i]['ca_coordinates'][:cutoff]
            batch[i]['side_chain_dihedrals'] = batch[i]['side_chain_dihedrals'][:cutoff]
            batch[i]['backbone_dihedrals'] = batch[i]['backbone_dihedrals'][:cutoff]
            batch[i]['tau_angle'] = batch[i]['tau_angle'][:cutoff]
            batch[i]['theta_angle'] = batch[i]['theta_angle'][:cutoff]
    
    sequences = [item['sequence'] for item in batch]
    ssequences = [item['sseq'] for item in batch]
    ca_coordinates = [item['ca_coordinates'] for item in batch]
    side_chain_dihedrals = [item['side_chain_dihedrals'] for item in batch]
    backbone_dihedrals = [item['backbone_dihedrals'] for item in batch]
    tau_angles = [item['tau_angle'] for item in batch]
    theta_angles = [item['theta_angle'] for item in batch]
    labels = [item['label'] for item in batch]

    # 找出序列中最长的长度，用于后续的padding
    max_length = max(len(seq) for seq in sequences)
    padded_ca_coordinates = []
    padded_side_chain_dihedrals = []
    padded_backbone_dihedrals = []
    padded_tau_angles = []
    padded_theta_angles = []
    padded_sequences = []
    padded_sseq = []
    padded_labels = []
    
    for label, seq, sseq, ca, sc_dih, bb_dih, tau, theta in zip(labels, sequences, ssequences, ca_coordinates, 
                                                   side_chain_dihedrals, backbone_dihedrals, tau_angles, theta_angles):
        ca = torch.tensor(ca, dtype=torch.float32)
        sc_dih = torch.tensor(sc_dih, dtype=torch.float32)
        bb_dih = torch.tensor(bb_dih, dtype=torch.float32)
        tau = torch.tensor(tau, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)

        if len(seq) == 0:
            return None
            continue
        if len(ca) > max_length:
            ca = ca[:max_length]
        if len(tau) > max_length:
            tau = tau[:max_length]
        if len(sc_dih) > max_length:
            sc_dih = sc_dih[:max_length]
        
        padding_ca = torch.cat([ca, torch.zeros((max_length - len(ca), 3))])
        padding_sc_dih = torch.cat([sc_dih, torch.zeros((max_length - len(sc_dih), 4))])
        padding_bb_dih = torch.cat([bb_dih, torch.zeros((max_length - len(bb_dih), 2))])
        padding_tau = torch.cat([tau, torch.zeros(max_length - len(tau))])
        padding_theta = torch.cat([theta, torch.zeros(max_length - len(theta))])
        
        padded_sequences.append(seq)
        padded_sseq.append(sseq)
        padded_ca_coordinates.append(padding_ca)
        padded_side_chain_dihedrals.append(padding_sc_dih)
        padded_backbone_dihedrals.append(padding_bb_dih)
        padded_tau_angles.append(padding_tau)
        padded_theta_angles.append(padding_theta)
        padded_labels.append(label)

    # 转换为Tensor，并使用stack进行堆叠，以形成批量
    padded_ca_coordinates = torch.stack(padded_ca_coordinates)
    padded_side_chain_dihedrals = torch.stack(padded_side_chain_dihedrals)
    padded_backbone_dihedrals = torch.stack(padded_backbone_dihedrals)
    padded_tau_angles = torch.stack(padded_tau_angles)
    padded_theta_angles = torch.stack(padded_theta_angles)
    if str(type(padded_labels[0])) == "<class 'torch.Tensor'>":
        labels = torch.stack(padded_labels)
    else:
        labels = torch.tensor(padded_labels)
    
    return {
        'sequence': padded_sequences,
        'sseq': padded_sseq,
        'ca_coordinates': padded_ca_coordinates,
        'side_chain_dihedrals': padded_side_chain_dihedrals,
        'backbone_dihedrals': padded_backbone_dihedrals,
        'tau_angle': padded_tau_angles,
        'theta_angle': padded_theta_angles,
        'label': labels
    }


def collate_protein_data(batch):
    if None in batch:
        print('dirty data')
        return None
    # batch是一个列表，其中每个元素都是__getitem__返回的一个样本
    cutoff = 1022
    for i in range(len(batch)):
        if len(batch[i]['sequence'])>cutoff:
            batch[i]['sequence'] = batch[i]['sequence'][:cutoff]
            batch[i]['ca_coordinates'] = batch[i]['ca_coordinates'][:cutoff]
            batch[i]['side_chain_dihedrals'] = batch[i]['side_chain_dihedrals'][:cutoff]
            batch[i]['backbone_dihedrals'] = batch[i]['backbone_dihedrals'][:cutoff]
            batch[i]['tau_angle'] = batch[i]['tau_angle'][:cutoff]
            batch[i]['theta_angle'] = batch[i]['theta_angle'][:cutoff]
    
    sequences = [item['sequence'] for item in batch]
    ca_coordinates = [item['ca_coordinates'] for item in batch]
    side_chain_dihedrals = [item['side_chain_dihedrals'] for item in batch]
    backbone_dihedrals = [item['backbone_dihedrals'] for item in batch]
    tau_angles = [item['tau_angle'] for item in batch]
    theta_angles = [item['theta_angle'] for item in batch]
    labels = [item['label'] for item in batch]

    # 找出序列中最长的长度，用于后续的padding
    max_length = max(len(seq) for seq in sequences)
    padded_ca_coordinates = []
    padded_side_chain_dihedrals = []
    padded_backbone_dihedrals = []
    padded_tau_angles = []
    padded_theta_angles = []
    padded_sequences = []
    padded_sseq = []
    padded_labels = []
    
    for label, seq, ca, sc_dih, bb_dih, tau, theta in zip(labels, sequences, ca_coordinates, 
                                                   side_chain_dihedrals, backbone_dihedrals, tau_angles, theta_angles):
        ca = torch.tensor(ca, dtype=torch.float32)
        sc_dih = torch.tensor(sc_dih, dtype=torch.float32)
        bb_dih = torch.tensor(bb_dih, dtype=torch.float32)
        tau = torch.tensor(tau, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)

        if len(seq) == 0:
            return None
            continue
        if len(ca) > max_length:
            ca = ca[:max_length]
        if len(tau) > max_length:
            tau = tau[:max_length]
        if len(sc_dih) > max_length:
            sc_dih = sc_dih[:max_length]
        
        padding_ca = torch.cat([ca, torch.zeros((max_length - len(ca), 3))])
        padding_sc_dih = torch.cat([sc_dih, torch.zeros((max_length - len(sc_dih), 4))])
        padding_bb_dih = torch.cat([bb_dih, torch.zeros((max_length - len(bb_dih), 2))])
        padding_tau = torch.cat([tau, torch.zeros(max_length - len(tau))])
        padding_theta = torch.cat([theta, torch.zeros(max_length - len(theta))])
        
        padded_sequences.append(seq)
        padded_ca_coordinates.append(padding_ca)
        padded_side_chain_dihedrals.append(padding_sc_dih)
        padded_backbone_dihedrals.append(padding_bb_dih)
        padded_tau_angles.append(padding_tau)
        padded_theta_angles.append(padding_theta)
        padded_labels.append(label)

    # 转换为Tensor，并使用stack进行堆叠，以形成批量
    padded_ca_coordinates = torch.stack(padded_ca_coordinates)
    padded_side_chain_dihedrals = torch.stack(padded_side_chain_dihedrals)
    padded_backbone_dihedrals = torch.stack(padded_backbone_dihedrals)
    padded_tau_angles = torch.stack(padded_tau_angles)
    padded_theta_angles = torch.stack(padded_theta_angles)
    if str(type(padded_labels[0])) == "<class 'torch.Tensor'>":
        labels = torch.stack(padded_labels)
    else:
        labels = torch.tensor(padded_labels)
    
    return {
        'sequence': padded_sequences,
        'ca_coordinates': padded_ca_coordinates,
        'side_chain_dihedrals': padded_side_chain_dihedrals,
        'backbone_dihedrals': padded_backbone_dihedrals,
        'tau_angle': padded_tau_angles,
        'theta_angle': padded_theta_angles,
        'label': labels
    }

class ProteinDataset_PPI(Dataset):
    """
    Dataset for Protein-Protein Interaction data processed by process_ppi.py.
    Loads data from a pickle file.
    """
    def __init__(self, cache_path):
        """
        Args:
            cache_path (str): Path to the .pkl file containing the processed data dictionary.
        """
        self.cache_path = cache_path
        self.data_cache = []

        print(f"Loading PPI data from {self.cache_path}...")
        try:
            with open(self.cache_path, 'rb') as f:
                # 加载由 process_ppi.py 保存的字典
                data_dic = pickle.load(f)
            # 将字典的值（每个样本的数据）转换为列表
            self.data_cache = list(data_dic.values())
            print(f"Loaded {len(self.data_cache)} PPI entries.")
        except FileNotFoundError:
            print(f"Error: Cache file not found at {self.cache_path}")
            # 你可以选择在这里抛出异常或者让 data_cache 保持为空
            # raise FileNotFoundError(f"Cache file not found at {self.cache_path}")
        except Exception as e:
            print(f"Error loading data from {self.cache_path}: {e}")
            # 处理其他可能的加载错误

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_cache)

    def __getitem__(self, idx):
        """
        Retrieves the sample data at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the data for the protein pair,
                  including 'protein1', 'protein2', 'plddt1', 'plddt2', 'label', 'key'.
        """
        return self.data_cache[idx]

if __name__ == '__main__':
    # dataset = ProteinDataset_Thermostability('./cache/Thermo/test.h5')
    #dataset = ProteinDataset_EC('./cache/EC/train_fix.h5')
    #dataset = ProteinDataset_DeepLoc('./datasets/DeepLoc/valid',
                                    #  './datasets/DeepLoc/valid.npy', c=2,
                                    #  cls2_npy_dir= './datasets/DeepLoc/valid.npy',
                                    #  cache_path='./SI-tuning/cache/deeploc/valid.h5')
    #dataset = ProteinDataset_Thermostability('/root/autodl-fs/datasets/Thermo/train.h5')
    # dataset = ProteinDataset_GO('/root/autodl-tmp/GO/valid','/root/autodl-tmp/GO',
    #                             cache_path='/root/autodl-fs/datasets/GO/valid')
    # dataset = ProteinDataset_GO('./cache/GO/train.h5')
    #dataset = ProteinDataset_MetalIonBinding(cache_path='./SI-tuning/cache/ION/train.h5')
                                            #  pdb_dir='./datasets/MetalIonBinding/train',
                                            #  npy_dir='./datasets/MetalIonBinding/train.npy',
                                            #  re_cache=True)
    #dataset = ProteinDataset_EC('/root/autodl-fs/datasets/EC/train','/root/autodl-tmp/datasets/EC/train', 
    #npy_dir='/root/autodl-tmp/datasets/EC/train.npy')
    # dataset = ProteinDataset_DeepLoc_cls10(pdb_dir='./datasets/DeepLoc/train', 
    #                                        npy_dir='./datasets/DeepLoc/train_cls10.npy', 
    #                                        cls2_npy_dir='./datasets/DeepLoc/train.npy', 
    #                                        cache_path='./cache/deeploc/train_cls10')
    # dataset = ProteinDataset_DeepLoc_cls10(pdb_dir='./datasets/DeepLoc/train_cls10', 
    #                                            npy_dir='./datasets/DeepLoc/train_cls10.npy', 
    #                                            cls2_npy_dir='./datasets/DeepLoc/train.npy', 
    #                                            cache_path='./cache/deeploc/train_cls10')
    #dataset = ProteinDataset_DeepLoc_2(cache_path='./cache/deeploc2.pkl', fold=0, train=True)
    dataset = ProteinDataset_PPI(cache_path='cache/HumanPPI/test.pkl')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_protein_data_sseq)
    for data in tqdm(dataloader):
        pass
    # plddt_means = []
    # for data in tqdm(dataloader):
    #     plddt = data['plddt']
    #     mean_plddt = plddt.mean().item()
    #     plddt_means.append(mean_plddt)
        
    # df = pd.DataFrame(plddt_means, columns=['Mean pLDDT'])

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df['Mean pLDDT'], bins=30, kde=True)
    # plt.title('Distribution of Mean pLDDT Values')
    # plt.xlabel('Mean pLDDT')
    # plt.ylabel('Frequency')
    # plt.grid(True)

    # # Save the figure as a PNG file
    # plt.savefig('mean_plddt_distribution.png')

