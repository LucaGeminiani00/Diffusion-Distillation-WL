import torch
from Utils.io_utils import instantiate_from_config


def create_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

if __name__ == '__main__':
    pass

