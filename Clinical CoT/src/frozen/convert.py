from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='src/frozen/checkpoints/epoch=0-step=0.ckpt')
    parser.add_argument('--output_path', type=str, default='src/frozen/checkpoints/epoch=0-step=0.ckpt')
    args = parser.parse_args()
    return args
    
def main(args):
    convert_zero_checkpoint_to_fp32_state_dict(args.input_path, args.output_path)

if __name__ == "__main__":
    args = get_args()
    main(args)