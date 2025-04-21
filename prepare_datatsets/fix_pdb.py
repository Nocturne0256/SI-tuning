import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_pdb_file(filename, input_directory, output_directory):
    try:
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.pdb")
        
        # 构建命令
        command = ['./PDB_Tool', '-i', input_file, '-R', '1', '-o', output_file]
        
        # 执行命令
        subprocess.run(command, check=True)
        return f"Successfully processed {filename}"
    except subprocess.CalledProcessError as e:
        return f"Error processing {filename}: {e}"

def main():
    # 设置输入和输出目录
    task = 'train_filtered_with_esm_cpu_esm_valid2'
    input_directory = f'./SI-tuning/lbs/{task}/pdb_files'
    output_directory = f'./SI-tuning/lbs/{task}/pdb_files_fix'

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 获取所有 PDB 文件
    pdb_files = [f for f in os.listdir(input_directory) if f.endswith('.pdb')]

    # 使用多进程进行处理
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_pdb_file, pdb_files, 
                                         [input_directory]*len(pdb_files), 
                                         [output_directory]*len(pdb_files)),
                            total=len(pdb_files), desc="Processing PDB files"))

    # 输出结果
    for result in results:
        print(result)

    print("All PDB files have been processed.")

if __name__ == "__main__":
    main()
