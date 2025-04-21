from Bio.PDB import PDBParser, Selection, calc_dihedral, PPBuilder, Chain
import warnings
from tqdm import tqdm
import pickle
from Bio.PDB import Polypeptide
from Bio.SeqUtils import seq1
import numpy as np

warnings.filterwarnings('ignore')

def get_dihedral_angles(residue):
    chi_angles = [0, 0, 0, 0]  # chi1, chi2, chi3, chi4 initialized to 0
    # Define atom names for each type of residue
    atom_names = {
        'ALA': [],
        'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
        'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
        'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
        'CYS': [['N', 'CA', 'CB', 'SG']],
        'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
        'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
        'GLY': [],
        'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
        'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
        'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
        'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
        'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
        'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
        'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
        'SER': [['N', 'CA', 'CB', 'OG']],
        'THR': [['N', 'CA', 'CB', 'OG1']],
        'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
        'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
        'VAL': [['N', 'CA', 'CB', 'CG1']],
    }
    res_name = residue.get_resname()
    if res_name in atom_names:
        for i, atoms in enumerate(atom_names[res_name]):
            try:
                atoms = [residue[x].get_vector() for x in atoms]
                angle = calc_dihedral(atoms[0],atoms[1],atoms[2],atoms[3])
                chi_angles[i] = angle
            except KeyError:
                pass  # Some atoms might be missing

    return chi_angles

def pdb_to_pkl(pdb_file, pkl_path=None, chain_id='A', parser = PDBParser()):
    structure = parser.get_structure('structure', pdb_file)
    if chain_id.islower():
        chain_id = chain_id.upper()
    if chain_id not in structure[0]:
        chain_id = chain_id.lower()
    if chain_id in structure[0]:
        chain = structure[0][chain_id]
        chain_residues = list(chain.get_residues())
    else:
        raise ValueError(f"Chain ID '{chain_id}' not found in the structure.")
    
    # 创建一个新的链对象来存储裁剪后的链
    chain = Chain.Chain("A")  # 创建一个新的链对象，链ID为"A"
    cnt = 0
    for residue in chain_residues:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            chain.add(residue)  # 添加裁剪后的残基到新链中
            cnt +=1
            if cnt == 1024:
                break

    sequence = ''
    ca_coordinates = []
    side_chain_dihedrals = []
    backbone_dihedrals = []
    tau_angle = []
    theta_angle = []

    for residue in chain:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            ca_atom = residue['CA']
            ca_coordinates.append(ca_atom.coord)
            side_chain_dihedrals.append(get_dihedral_angles(residue))
            #sequence+=residue.get_resname()[0]
        
    # Append the backbone dihedral angles
    polypeptides = PPBuilder().build_peptides(chain)
    for poly_index, poly in enumerate(polypeptides):
        phi_psi = poly.get_phi_psi_list()
        for res_index, (phi, psi) in enumerate(phi_psi):
            if phi and psi:  #忽略没有phi或psi角的残基
                backbone_dihedrals.append((phi, psi))
            else:
                backbone_dihedrals.append((0, 0))

        tau = poly.get_tau_list()
        for res_index, (angle) in enumerate(tau):
            if angle:  
                tau_angle.append(angle)
            else:
                tau_angle.append(0)
        tau_angle += [0,0,0]
        
        theta = poly.get_theta_list()
        for res_index, (angle) in enumerate(theta):
            if angle:  
                theta_angle.append(angle)
            else:
                theta_angle.append(0)
        theta_angle += [0,0]

        sequence += str(poly.get_sequence())

    #ca_coordinates_array = np.array(ca_coordinates)
    #distance_matrix = np.sqrt(np.sum((ca_coordinates_array[:, None] - ca_coordinates_array)**2, axis=-1))
    len_seq = len(sequence)
    len_tau = len(tau_angle)
    len_ca = len(ca_coordinates)
    if len_tau < len_seq:
        tau_angle += [0]*(len_ca-len_tau)
        theta_angle += [0]*(len_ca-len_tau)
    elif len_tau > len_seq:
        tau_angle = tau_angle[:len_ca]
        theta_angle = theta_angle[:len_ca]
    if len_ca > len_seq:
        ca_coordinates = ca_coordinates[:len_seq]
        side_chain_dihedrals = side_chain_dihedrals[:len_seq]
    #assert len(tau_angle) == len_ca
    protein_data = {
        'sequence': sequence, #N
        'ca_coordinates': ca_coordinates, # N * 3
        'side_chain_dihedrals': side_chain_dihedrals, # N * 4, [-pi, pi]
        'backbone_dihedrals': backbone_dihedrals, # N * 2, [-pi, pi]
        #'distance_matrix': distance_matrix, # N * N ndarray
        'tau_angle': tau_angle, # N, [-pi, pi]
        'theta_angle': theta_angle # N, [-pi, pi]
    }
    #assert len(ca_coordinates) == len(sequence)
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(protein_data, f)
    return protein_data
# 使用例
#pdb_to_pkl('/root/autodl-tmp/datasets/EC/train/1w6s.pdb')

