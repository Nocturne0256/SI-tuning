import os
from utils import *
from torch.utils.data import DataLoader, Dataset
from SI_tuning import SI_tuning
from tasks import *
import argparse
import yaml
from pathlib import Path

def initialize_logger(task, model_size, log_file, b_angle, b_coord, fusion_dim, lora_r, train_batch_size, lr):
    """
    Initialize the logger for the training process.
    """
    if log_file=='empty':
        log_file = './log/' + task + '_' + model_size
    else:
        log_file = './log/' + task + '_' + log_file
    if not b_angle:
        log_file += '_no_angle'
    if not b_coord:
        log_file += '_no_coord'

    warnings.filterwarnings('ignore')
    logging.getLogger('MDAnalysis').setLevel(logging.WARNING)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s', force=True)

    logging.info('task:%s, lr:%.9f, train_batch_size:%d, model_size:%s, b_angle:%d, b_coord%d, fusion_dim:%d, lora_r:%d' 
                 %(task, lr, train_batch_size, model_size, b_angle, b_coord, fusion_dim, lora_r))
    print(task, lr, train_batch_size, model_size, log_file, fusion_dim, b_angle, b_coord)

def train_epoch(model, dataloader, optimizer, metrics, tokenizer, num_labels, stack_labels, loss_batch, epoch, plddt=False):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    running_gamma = 0.0
    rets, labels = stack_labels(num_labels)
    for i, inputs in tqdm(enumerate(dataloader), total = len(dataloader)): 
        if plddt:
            inps = [inputs['side_chain_dihedrals'].cuda(), inputs['backbone_dihedrals'].cuda(), 
                    inputs['tau_angle'].cuda(), inputs['theta_angle'].cuda(), inputs['plddt'].cuda()]
        else:
            inps = [inputs['side_chain_dihedrals'].cuda(), inputs['backbone_dihedrals'].cuda(), 
                    inputs['tau_angle'].cuda(), inputs['theta_angle'].cuda()]
        label = inputs['label'].cuda()
        coords = inputs['ca_coordinates'].cuda()
        seq = tokenizer(inputs['sequence'], return_tensors="pt", padding=True)
        ret, loss, gamma = model(inps, seq['input_ids'].cuda(), seq['attention_mask'].cuda(), label, coords, plddt)
        loss = loss.mean()
        gamma = gamma.mean()
        rets = torch.cat([rets, ret.detach().cpu()], dim=0)
        labels = torch.cat([labels, label.detach().cpu()], dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        running_gamma += gamma.item()
        if (i+1) % loss_batch == 0:
            logging.info('[%d, %5d] loss: %.5f lr: %.9f gamma: %6f '%
                    (epoch + 1, (i+1), running_loss / loss_batch, optimizer.state_dict()['param_groups'][0]['lr'], running_gamma/loss_batch))
            running_loss = 0.0
            running_gamma = 0.0
    
    train_acc =  metrics(rets, labels)
    logging.info('epoch:%d, train_loss:%.5f, train_metrics:%.5f'%(epoch+1, epoch_loss/len(dataloader), train_acc))
    return epoch_loss

def valid_epoch(model, dataloader, metrics, tokenizer, num_labels, stack_labels, plddt=False, valid='valid'):
    """
    Validate the model on the validation set or test set.
    """
    with torch.no_grad(): 
        model.eval()
        loss = 0.0
        rets, labels = stack_labels(num_labels)
        
        for i, inputs in tqdm(enumerate(dataloader), total = len(dataloader)):
            if plddt:
                inps = [inputs['side_chain_dihedrals'].cuda(), inputs['backbone_dihedrals'].cuda(), 
                        inputs['tau_angle'].cuda(), inputs['theta_angle'].cuda(), inputs['plddt'].cuda()]
            else:
                inps = [inputs['side_chain_dihedrals'].cuda(), inputs['backbone_dihedrals'].cuda(), 
                        inputs['tau_angle'].cuda(), inputs['theta_angle'].cuda()]
            label = inputs['label'].cuda()
            coords = inputs['ca_coordinates'].cuda()
            seq = tokenizer(inputs['sequence'], return_tensors="pt", padding=True)
            ret, loss, gamma = model(inps, seq['input_ids'].cuda(), seq['attention_mask'].cuda(), label, coords, plddt)
            loss = loss.mean()
            rets = torch.cat([rets, ret.detach().cpu()], dim=0)
            labels = torch.cat([labels, label.detach().cpu()], dim=0)
            loss += loss.item()
        
        acc = metrics(rets, labels)
        logging.info('Testing on %s set: metrics:%.9f, loss:%.9f'%(valid, acc, loss/len(dataloader)))
    return acc

def finetune(task='deeploc', lr=(0.001)/16, train_batch_size = 1, epochs=200,
             model_size = '650', log_file='empty', 
             fusion_dim=16, b_angle=True, b_coord=True, ckpt=None, lora_r = 16, esm_path=None, plddt=False):
    
    initialize_logger(task, model_size, log_file, b_angle, b_coord, fusion_dim, lora_r, train_batch_size, lr)

    num_labels, loss_fn, dataset, valid_dataset, test_dataset, metrics, task_type, stack_labels = get_task(task)
    model = SI_tuning(num_labels = num_labels, loss_fn = loss_fn, config_path=esm_path, model_size=model_size, 
                      task=task_type, fusion_dim=fusion_dim, plddt=plddt, b_angle=b_angle, b_coord=b_coord, lora_r= lora_r)
    model.cuda()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    logging.info('Train_data:%d, Valid_data:%d, Test_data:%d, bz:%d' % (len(dataset), len(valid_dataset), len(test_dataset), train_batch_size))
    if plddt:
        collate_fn = collate_protein_data_plddt
    else:
        collate_fn = collate_protein_data
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn) 

    tokenizer =  AutoTokenizer.from_pretrained(esm_path)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25,min_lr=0.00000005, patience = 10)
    
    loss_batch = 100
    best_valid_acc = 0
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, dataloader, optimizer, metrics, tokenizer, num_labels, stack_labels, loss_batch, epoch, plddt=plddt)
        valid_acc = valid_epoch(model, valid_dataloader, metrics, tokenizer, num_labels, stack_labels, plddt=plddt, valid='valid')
        _ = valid_epoch(model, test_dataloader, metrics, tokenizer, num_labels, stack_labels, plddt=plddt, valid='test')

        scheduler.step(epoch_loss)
        
        if ckpt != 'empty':
            save_checkpoint(epoch, model, optimizer, f'checkpoints/{ckpt}_last.pth')
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                save_checkpoint(epoch, model, optimizer, f'checkpoints/{ckpt}_best.pth')

if __name__ == '__main__':
    setup_seed(2024)
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Set GPU device
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get('gpu', 0))
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    #print('GPU:', os.environ["CUDA_VISIBLE_DEVICES"])

    
    print('training with angle and coord:', cfg.get('no_angle', True), cfg.get('no_coord', True))
    
    # Call finetune function with config parameters
    try:
        finetune(
            task=cfg['task'],
            lr=cfg.get('lr', 0.00002/8),
            train_batch_size=cfg.get('train_batch_size', 1),
            epochs=cfg.get('epochs', 200),
            model_size=cfg.get('model_size', '650'),
            log_file=cfg.get('log', 'empty'),
            fusion_dim=cfg.get('fusion_dim', 768),
            b_angle=not cfg.get('no_angle', True),
            b_coord=not cfg.get('no_coord', True),
            ckpt=cfg.get('ckpt', 'empty'),
            lora_r=cfg.get('lora_r', 16),
            esm_path=cfg.get('esm_path', None),
            plddt=cfg.get('plddt', False)
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        # 记录错误信息到日志文件
        logging.error(f"error: {e}")
        raise e
