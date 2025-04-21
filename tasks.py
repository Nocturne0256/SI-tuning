from utils import *
from torch.utils.data import DataLoader, Dataset
from datasets import *
def get_task(task='deeploc'):
    if task=='deeploc':
        num_labels = 1
        loss_fn = 'BCE'
        dataset = ProteinDataset_DeepLoc(cache_path='./cache/deeploc/train.h5')
        valid_dataset = ProteinDataset_DeepLoc(cache_path='./cache/deeploc/valid.h5')
        test_dataset = ProteinDataset_DeepLoc(cache_path='./cache/deeploc/test.h5')
        metrics = calculate_accuracy
        task_type = 'classification'
        stack_labels = stack_labels_

    elif task=='cls10':
        num_labels = 10
        loss_fn = 'CE'
        dataset = ProteinDataset_DeepLoc_cls10(pdb_dir='./datasets/DeepLoc/train_cls10', 
                                               npy_dir='./datasets/DeepLoc/train_cls10.npy', 
                                               cls2_npy_dir='./datasets/DeepLoc/train.npy', 
                                               cache_path='./cache/deeploc/train_cls10')
        valid_dataset = ProteinDataset_DeepLoc_cls10(pdb_dir='./datasets/DeepLoc/valid_cls10', 
                                                    npy_dir='./datasets/DeepLoc/valid_cls10.npy', 
                                                    cls2_npy_dir='./datasets/DeepLoc/valid.npy', 
                                                    cache_path='./cache/deeploc/valid_cls10')
        test_dataset = ProteinDataset_DeepLoc_cls10(pdb_dir='./datasets/DeepLoc/test_cls10', 
                                                    npy_dir='./datasets/DeepLoc/test_cls10.npy', 
                                                    cls2_npy_dir='./datasets/DeepLoc/test.npy', 
                                                    cache_path='./cache/deeploc/test_cls10')
        metrics = calculate_accuracy_cls10
        task_type = 'classification'
        stack_labels = stack_labels_cls10

    elif task=='deeploc2':
        num_labels = 11
        loss_fn = 'CE'
        dataset = ProteinDataset_DeepLoc_2(cache_path='./cache/deeploc2.pkl', fold=0, train=True)
        valid_dataset = ProteinDataset_DeepLoc_2(cache_path='./cache/deeploc2.pkl', fold=0, train=False)
        test_dataset = ProteinDataset_DeepLoc_2(cache_path='./cache/deeploc2.pkl', fold=0, train=False)
        metrics = calculate_accuracy_cls10
        task_type = 'classification'
        stack_labels = stack_labels_cls10
        
    elif task=='thermo':
        num_labels = 1
        loss_fn = 'MSE'
        dataset = ProteinDataset_Thermostability('./cache/Thermo/train.h5')
        valid_dataset = ProteinDataset_Thermostability('./cache/Thermo/valid.h5')
        test_dataset = ProteinDataset_Thermostability('./cache/Thermo/test.h5')
        metrics = calculate_spearman
        task_type = 'regression'
        stack_labels = stack_labels_

    elif task=='EC':
        num_labels = 585
        loss_fn = 'BCE'
        dataset = ProteinDataset_EC('./cache/EC/train_fix.h5')
        valid_dataset = ProteinDataset_EC('./cache/EC/valid_fix.h5')
        test_dataset = ProteinDataset_EC('./cache/EC/test_fix.h5')
        metrics = f1_max
        task_type = 'classification'
        stack_labels = stack_labels_

    # elif task =='lbs':
    #     num_labels = 37
    #     loss_fn = 'BCE'
    #     dataset = LBSDataset('./SI-tuning/lbs/train_filtered_with_esm_cpu_esm_train2')
    #     valid_dataset = LBSDataset('./SI-tuning/lbs/train_filtered_with_esm_cpu_esm_valid2')
    #     test_dataset = LBSDataset('./SI-tuning/lbs/ASTEX85_with_esm_cpu2')
    #     metrics = f1_max
    #     task_type = 'lbs'
    #     stack_labels = stack_labels_

    elif task=='ion':
        num_labels = 1
        loss_fn = 'BCE'
        dataset = ProteinDataset_MetalIonBinding(cache_path='./cache/ION/train.h5')
        valid_dataset = ProteinDataset_MetalIonBinding(cache_path='./cache/ION/valid.h5')
        test_dataset = ProteinDataset_MetalIonBinding(cache_path='./cache/ION/test.h5')
        metrics = calculate_accuracy
        task_type = 'classification'
        stack_labels = stack_labels_
    
    else:
        print("task "+ task + " does not exist!")

    return num_labels, loss_fn, dataset, valid_dataset, test_dataset, metrics, task_type, stack_labels

def stack_labels_(num_labels=1):
    if num_labels == 1:
        rets = torch.empty((0))
        labels = torch.empty((0))
    else:
        rets = torch.empty((0, num_labels))
        labels = torch.empty((0, num_labels))
    return rets, labels

def stack_labels_cls10(num_labels=1):
    rets = torch.empty((0, num_labels))
    labels = torch.empty((0))
    return rets, labels