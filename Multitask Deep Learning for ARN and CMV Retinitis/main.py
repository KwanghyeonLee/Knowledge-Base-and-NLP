import os, gc
import numpy as np
import torch


from torch.utils.data import DataLoader
from src.dataset import AdversarialDataset
from src.model import SingleTaskMLP, FullySharedMTL, SharedPrivateMTL, AdversarialMTL
from src.Trainer import Trainer

from src.utils.utils import fix_random_seed, make_logger, logger_result
from src.utils.option import get_opt


def main(opt):
    # Load dataset & dataloader
    train_loader = None
    valid_loader = None
    if opt['task'] == 'train':
        train_data = AdversarialDataset(mode = 'train')
        valid_data = AdversarialDataset(mode = 'valid')
        train_loader = DataLoader(train_data, batch_size = opt['batch_size'], shuffle = True)
        valid_loader = DataLoader(valid_data, batch_size = opt['batch_size'], shuffle = False)
        opt['logger'].info('Train & Valid data loaded')
        
    test_data = AdversarialDataset(mode = 'test')
    test_loader = DataLoader(test_data, batch_size = opt['batch_size'], shuffle = False)
    opt['logger'].info('Test data loaded')
    
    # Get model
    if opt['model_name'] == 'STL':
        model = SingleTaskMLP(hidden_unit = opt['hidden_unit'], class_type = opt['class_type'])
    elif opt['model_name'] == 'FSMTL':
        model = FullySharedMTL(hidden_unit = opt['hidden_unit'])
    elif opt['model_name'] == 'SPMTL':
        model = SharedPrivateMTL(hidden_unit = opt['hidden_unit'])
    else:
        model = AdversarialMTL(hidden_unit = opt['hidden_unit'])
    opt['logger'].info('Model initialized')
    
    # get model path
    opt['model_path'] = os.path.join(opt['model_dir'], opt['model_name'], 'smote')
    opt['model_path'] = opt['model_path'] + ('_' + str(opt['class_type']) + '.pth' if opt['model_name'] == 'STL' else '.pth')
    
    # Load model (if the task is not train)
    if opt['task'] != 'train':
        model.load_state_dict(torch.load(opt['model_path'], map_location = opt['device']))
        opt['logger'].info('Model loaded')
    
    # Get trainer
    trainer = Trainer(opt, model, train_loader, valid_loader, test_loader)
    opt['logger'].info('Trainer initialized')
    
    # Train
    if opt['task'] == 'train':
        opt['logger'].info('Training started')
        trainer.train()
        opt['logger'].info('Training finished')
    
    # Test
    if opt['do_test']:
        opt['logger'].info('Testing started')
        avg_losses, auc_scores, labels, preds = trainer.evaluate(mode='test')
        
        # Log result
        logger_result(opt, 'test', avg_losses, auc_scores)
    
    return 0


if __name__ == '__main__':
    
    # get options (args)
    opt = get_opt()
    
    # Get logger 
    opt['logger'] = make_logger(opt['logger_dir'], opt['model_name'])
    
    
    # clear memory
    gc.collect()
    torch.cuda.empty_cache()
    opt['logger'].info('Memory cleared')
    
    # get device
    # if opt['device_cpu']:
    #     device = torch.device('cpu')
    # else:
    device = torch.device('cuda' if torch.cuda.is_available() and not opt['device_cpu'] else 'cpu')
    print(device)
    opt["device"] = device
    opt['logger'].info('Device: {}'.format(device))
    
    # set random seed
    fix_random_seed(opt['random_seed'])
    opt['logger'].info('Random seed fixed to {}'.format(opt['random_seed']))
    
    main(opt)