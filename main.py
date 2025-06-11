import os
import torch
import argparse
import numpy as np

from Engine.logger import Logger
from Engine.trainer import Engine
from Datasets.create_dataloader import create_dataloader
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()

    dataloader_info = create_dataloader(config, args)
    trainer = Engine(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()


#(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
