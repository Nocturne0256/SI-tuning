from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
# tqdm 重复导入，移除一个
# from tqdm import tqdm
from pdb_to_pkl import pdb_to_pkl
import h5py
import pickle


warnings.filterwarnings('ignore')

# 修改 process_key 以处理两个蛋白质
def process_key(key, data_entry, pdb_dir):
    name_1 = data_entry['name_1']
    name_2 = data_entry['name_2']

    pdb_file1 = os.path.join(pdb_dir, f"{name_1}.pdb")
    pdb_file2 = os.path.join(pdb_dir, f"{name_2}.pdb")

    # 检查 PDB 文件是否存在 (虽然在 process_ppi 中已检查，双重检查更安全)
    if not os.path.exists(pdb_file1) or not os.path.exists(pdb_file2):
        print(f"Warning: PDB file not found for key {key}. Skipping.")
        return None

    try:
        protein_data1 = pdb_to_pkl(pdb_file1, 'A')
        protein_data2 = pdb_to_pkl(pdb_file2, 'A')
    except Exception as e:
        print(f"Error processing PDB for key {key}: {e}")
        return None

    # 检查 pdb_to_pkl 是否成功返回数据并且包含序列信息
    if protein_data1 is None or protein_data1.get('sequence') is None:
        print(f"Error: Failed to process protein 1 ({name_1}) for key {key}.")
        return None
    if protein_data2 is None or protein_data2.get('sequence') is None:
        print(f"Error: Failed to process protein 2 ({name_2}) for key {key}.")
        return None

    # 提取 fitness 和 plddt
    targets = torch.tensor(data_entry['label'], dtype=torch.float32)
    plddt1 = torch.tensor(data_entry['plddt_1'], dtype=torch.float32)
    plddt2 = torch.tensor(data_entry['plddt_2'], dtype=torch.float32)

    # 组合结果
    result_data = {
        'protein1': protein_data1,
        'protein2': protein_data2,
        'plddt1': plddt1,
        'plddt2': plddt2,
        'label': targets,
        'key': key # 保留原始 key
    }

    return key, result_data

# 修改 process_ppi 以适应新的数据结构和处理逻辑
def process_ppi(cache_path, pdb_dir, npy_dir):
    # data_cache = [] # 不再需要这个列表
    dic = {}

    print(f"Loading data from {npy_dir}...")
    data = np.load(npy_dir, allow_pickle=True).item()
    print(f"Loaded {len(data)} entries.")

    print("Filtering keys based on PDB file existence...")
    keys_to_process = []
    for key, entry in tqdm(data.items(), desc="Checking PDB files"):
        # 检查 entry 是否包含必要的键
        if not all(k in entry for k in ['name_1', 'name_2', 'plddt_1', 'plddt_2', 'label']):
            print(f"Warning: Skipping key {key} due to missing required fields (name_1, name_2, plddt1, plddt2, fitness).")
            continue

        pdb_file1_path = os.path.join(pdb_dir, f"{entry['name_1']}.pdb")
        pdb_file2_path = os.path.join(pdb_dir, f"{entry['name_2']}.pdb")

        if os.path.exists(pdb_file1_path) and os.path.exists(pdb_file2_path):
            keys_to_process.append(key)
        # else:
            # print(f"Debug: Skipping key {key}. PDB missing: {pdb_file1_path if not os.path.exists(pdb_file1_path) else ''} {pdb_file2_path if not os.path.exists(pdb_file2_path) else ''}")

    print(f"Found {len(keys_to_process)} entries with existing PDB files.")

    if not keys_to_process:
        print("No valid entries found to process. Exiting.")
        return

    print("Processing entries...")
    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor() as executor:
        # 提交任务，传递 key, data[key], pdb_dir 给 process_key
        futures = {executor.submit(process_key, key, data[key], pdb_dir): key for key in keys_to_process}
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PPI data"):
            try:
                result = future.result()
                if result is not None:
                    res_key, protein_data = result
                    dic[res_key] = protein_data
            except Exception as e:
                # 捕获子进程中的异常
                key_error = futures[future] # 获取导致错误的 key
                print(f"Error processing key {key_error} in subprocess: {e}")


    print(f"Successfully processed {len(dic)} entries.")
    if dic:
        print(f"Saving processed data to {cache_path}...")
        # 使用pickle保存数据字典
        with open(cache_path, 'wb') as f:
            pickle.dump(dic, f)
        print("✅ Data saved successfully.")
    else:
        print("No data was processed successfully.")

if __name__ == '__main__':
    base_data_dir = r'./HumanPPI' # 使用原始字符串或双反斜杠
    base_cache_dir = r'./SI-tuning/cache/HumanPPI' # 假设缓存目录结构

    # 创建缓存目录（如果不存在）
    os.makedirs(base_cache_dir, exist_ok=True)

    # 处理 test 数据集
    print("\nProcessing Test Set...")
    process_ppi(cache_path=os.path.join(base_cache_dir, 'test.pkl'),
                   pdb_dir=os.path.join(base_data_dir, 'test'),
                   npy_dir=os.path.join(base_data_dir, 'test.npy'))

    print("\nProcessing Train Set...")
    process_ppi(cache_path=os.path.join(base_cache_dir, 'train.h5'),
                   pdb_dir=os.path.join(base_data_dir, 'train'), # 假设目录名为 train
                   npy_dir=os.path.join(base_data_dir, 'train.npy')) # 假设 npy 文件名为 train.npy

    print("\nProcessing Validation Set...")
    process_ppi(cache_path=os.path.join(base_cache_dir, 'valid.h5'),
                   pdb_dir=os.path.join(base_data_dir, 'valid'), # 假设目录名为 valid
                   npy_dir=os.path.join(base_data_dir, 'valid.npy')) # 假设 npy 文件名为 valid.npy
