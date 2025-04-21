import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO

def split_chain(pdb_file, input_dir, output_dir):
    try:
        parser = PDBParser(QUIET=True)
        io = PDBIO()
        pdb_id = pdb_file[:-4]
        structure = parser.get_structure(pdb_id, os.path.join(input_dir, pdb_file))
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                output_filename = f"{pdb_id}_{chain_id}.pdb"
                output_path = os.path.join(output_dir, output_filename)
                
                io.set_structure(chain)
                io.save(output_path)
        return f"Successfully processed {pdb_file}"
    except Exception as e:
        return f"Error processing {pdb_file}: {e}"

def split_pdb_by_chain(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(split_chain, pdb_files, 
                                         [input_dir]*len(pdb_files), 
                                         [output_dir]*len(pdb_files)),
                            total=len(pdb_files), desc="Splitting PDB files"))

    print('success')

# 示例用法
input_directorys = ['./datasets/EC/train', './datasets/EC/test',
                     './datasets/EC/valid',
                     './datasets/GO/train', './datasets/GO/test',
                    './datasets/GO/valid',
                    './datasets/MetalIonBinding/train', 
                    './datasets/MetalIonBinding/test',
                    './datasets/MetalIonBinding/valid']
                    # '/mnt/e/datasets/MetalIonBinding/train', '/mnt/e/datasets/MetalIonBinding/test', '/mnt/e/datasets/MetalIonBinding/valid']


for input_dir in input_directorys:
    print('processing', input_dir)
    split_pdb_by_chain(input_dir, input_dir+'_chain')
