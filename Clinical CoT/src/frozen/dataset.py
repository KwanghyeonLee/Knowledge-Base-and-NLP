import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Dict, Sequence
import transformers
from typing import List, Union
from dataclasses import dataclass, field
IGNORE_INDEX = -100

def get_dataloader(tokenizer, args, **kwargs) -> List[Union[DataLoader, None]]:
    # Initialize your datasets
    if args.do_train:
        if args.train_wo_rationale:
            train_dataset = ClinicalReasoningDatasetwithoutRationale(mode='train', mri=args.img_type, data_portion=args.data_portion)
            valid_dataset = ClinicalReasoningDatasetwithoutRationale(mode='validation', mri=args.img_type, data_portion=args.data_portion)
        else:
            train_dataset = ClinicalReasoningDataset(mode='train', mri=args.img_type, data_portion=args.data_portion)
            valid_dataset = ClinicalReasoningDataset(mode='validation', mri=args.img_type, data_portion=args.data_portion)
    if args.do_test:
        if args.train_wo_rationale:
            test_dataset = ClinicalReasoningDatasetwithoutRationale(mode='test', mri=args.img_type, data_portion=args.data_portion)
            test_AIBL_dataset = ClinicalReasoningDatasetwithoutRationale(mode='test_AIBL', mri=args.img_type, data_portion=args.data_portion)
        else:
            test_dataset = ClinicalReasoningDataset(mode='test', mri=args.img_type, data_portion=args.data_portion)
            test_AIBL_dataset = ClinicalReasoningDataset(mode='test_AIBL', mri=args.img_type, data_portion=args.data_portion)
    
    # Initialize your DataCollatorForFrozen
    data_collator = DataCollatorForFrozen(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        img_token_num=args.img_token_num,
        predict_with_generate=args.predict_with_generate,
    )
    data_collator_test = DataCollatorForFrozen(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        img_token_num=args.img_token_num,
        predict_with_generate=True,
    )
    
    # Initialize your DataLoaders
    return_list = []
    if args.do_train:
        train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator, num_workers=32, **kwargs)
        valid_loader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator, num_workers=32, **kwargs)
        return_list.append(train_loader)
        return_list.append(valid_loader)
    if args.do_test:
        test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator_test, num_workers=32, **kwargs)
        return_list.append(test_loader)
        test_AIBL_dataset = DataLoader(test_AIBL_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator_test, num_workers=32, **kwargs)
        return_list.append(test_AIBL_dataset)
        
    return return_list


@dataclass
class DataCollatorForFrozen(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    img_token_num: int
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        input_img = torch.stack([example['image'] for example in instances])
        sources = [example['input'] for example in instances]
        targets = [f"{example['label']} {self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        
        labels = [] 
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'], 
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source) + self.img_token_num)] + copy.deepcopy(tokenized_target))
                )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if not self.predict_with_generate:
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        new_tokens_mask = torch.ones((input_ids.size(0), self.img_token_num), dtype=torch.bool)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        new_attention_mask = torch.cat((attention_mask, new_tokens_mask), dim=1)
        
        data_dict = {
            'input_img': input_img,
            'input_ids': input_ids,
            'attention_mask':new_attention_mask,
        }
        if labels != []:
            data_dict['labels'] = labels
        return data_dict

class ClinicalReasoningDataset(Dataset):
    def __init__(self, mode='train', mri='mr', data_portion='all', **kwargs):
        '''
            mode: 'train', 'validatoin', 'test', 'test_AIBL
        '''
        if data_portion == 'all':
            root_dir = 'dir/to/data'
        else:
            root_dir = os.path.join("dir/to/split/root/data", data_portion)
        
        self.mode = mode
        self.mri = mri
        self.data_dir = os.path.join(root_dir, self.mode)
        self.data_list = sorted(os.listdir(self.data_dir), key=lambda x: int(
            x.split('_')[-1].split('.')[0]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.data_list[idx]),allow_pickle=True)

        # assert that the data is not None
        assert(data is not None)

        # Process Image
        if self.mri == 'mr':
            image = torch.from_numpy(data['image_mr'])
        elif self.mri == 'parcel':
            image = torch.from_numpy(data['image_parcel'])

        return {
            'image': image,
            'input': str(data['input']),
            'label': str(data['label']),
        }
        
class ClinicalReasoningDatasetwithoutRationale(Dataset):
    def __init__(self, mode='train', mri='mr', data_portion='all', **kwargs):
        '''
            mode: 'train', 'validatoin', 'test', 'test_AIBL
        '''
        if data_portion == 'all':
            root_dir = 'dir/to/data'
        else:
            root_dir = os.path.join("dir/to/split/root/data", data_portion)
            
        self.mode = mode
        self.mri = mri
        self.data_dir = os.path.join(root_dir, self.mode)
        self.data_list = sorted(os.listdir(self.data_dir), key=lambda x: int(
            x.split('_')[-1].split('.')[0]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.data_list[idx]),allow_pickle=True)

        # assert that the data is not None
        assert(data is not None)

        # Process Image
        if self.mri == 'mr':
            image = torch.from_numpy(data['image_mr'])
        elif self.mri == 'parcel':
            image = torch.from_numpy(data['image_parcel'])
        label_text = str(data['label'])
        splited_text = label_text.split('Answer:')
        answer_without_rationale = f"Answer:{splited_text[1]}"
        return {
            'image': image,
            'input': str(data['input']),
            'label': answer_without_rationale,
        }