# add python path for safety
import sys
sys.path.append('dir/to/your/project')

import os
import json
import yaml
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import transformers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.frozen.dataclass import ModelArguments, DataArguments, TrainingArguments, GenerationArguments
from src.frozen.dataset import get_dataloader
from src.frozen.VLTrainer import VisionLanguageModelTrainer
from src.frozen.VLmodel import Frozen

import deepspeed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/frozen/config/inference.yaml')
    parser.add_argument('--config_name', type=str, default='inference_parcel_resnet152')
    parser.add_argument('--train_wo_rationale', action='store_true')
    parser.add_argument('--data_portion', choices=['all', '75', '50', '25'], default='all')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()
    
    return args

def main(args):
    # open config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[args.config_name]
    
    
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args = hfparser.parse_dict(config)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(generation_args), **vars(args)
    )
    

    # get Logger & config
    wandb_logger = WandbLogger(project=args.wandb_project,
                               name=args.wandb_name, save_dir=args.save_dir)
    
    # Initialize the model & Trainer
    print("Loading model")
    frozen_model = Frozen(model_args)
    model = VisionLanguageModelTrainer(frozen_model, args)
    print("Loaded model")
    if args.pretrained_path is not None:
        print("Loading pretrained model from {}".format(args.pretrained_path))
        model.load_state_dict(torch.load(args.pretrained_path)['state_dict'])
        print("Loaded pretrained model from {}".format(args.pretrained_path))
    try:
        tokenizer = model.model.get_tokenizer()
    except:
        tokenizer = model.get_tokenizer()
        
    # Set ModelCheckpoint
    callback_list = get_ModelCheckpoint()
    
    print("Start Initializing Trainer")
    
    trainer = pl.Trainer(callbacks=callback_list, devices=args.devices, accelerator='gpu',
                         max_epochs=args.max_epochs, logger=wandb_logger, strategy=args.strategy, precision=16)
    
    print("Initialized Trainer")
    data_loaders = get_dataloader(tokenizer, args)
    
    # Train the model
    if args.do_train:
        train_dataloader, val_dataloader = data_loaders[:2]
        # assert train and validation dataloader is not None
        assert(train_dataloader is not None)
        assert(val_dataloader is not None)

        trainer.fit(model, train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)
        
        if args.do_test:
            test_dataloader = data_loaders[-1]
            # assert test dataloader is not None
            assert(test_dataloader is not None)            
            
            trainer.test(model, test_dataloader, ckpt_path="best")

    # Test the model
    if not args.do_train and args.do_test:
        test_dataloader = data_loaders[-2]
        # assert test dataloader is not None
        assert(test_dataloader is not None)            
            
        trainer.test(model, test_dataloader, ckpt_path="last")
        
        test_AIBL_dataloader = data_loaders[-1]
        assert(test_AIBL_dataloader is not None)
        trainer.test(model, test_AIBL_dataloader, ckpt_path="last")
        
    return None

def get_ModelCheckpoint():
    val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='val_loss-{epoch:02d}-{val_loss:.2f}',
        save_top_k=4,
        mode='min',
    )
    return [val_checkpoint]


if __name__ == '__main__':
    # Empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # fix seed
    pl.seed_everything(42)

    # get args
    args = get_args()
    # start main
    main(args)
