import argparse
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch_warmup_lr import WarmupLR

from dlcv.config import get_cfg_defaults, CN
from dlcv.model import fasterrcnn_model, fcos_model, retinanet_model
from dlcv.utils import write_results_to_csv, save_model, custom_collate_fn, get_transforms
from dlcv.training import train_and_evaluate_model
from dlcv.dataset1 import Dataset1

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def main(cfg, args):
    if args.distributed:
        init_distributed_mode(args)

    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.TRAIN.NO_CUDA else 'cpu')

    transform_train = get_transforms(train=True, 
                                     image_height=cfg.TRAIN.IMAGE_HEIGHT, 
                                     image_width=cfg.TRAIN.IMAGE_WIDTH, 
                                     horizontal_flip_prob=cfg.TRAIN.HORIZONTAL_FLIP_PROB, 
                                     rotation_degrees=cfg.TRAIN.ROTATION_DEGREES)
    
    transform_val = get_transforms(train=False, 
                                     image_height=cfg.TRAIN.IMAGE_HEIGHT, 
                                     image_width=cfg.TRAIN.IMAGE_WIDTH, 
                                     horizontal_flip_prob=cfg.TRAIN.HORIZONTAL_FLIP_PROB, 
                                     rotation_degrees=cfg.TRAIN.ROTATION_DEGREES)
                                     
    train_dataset = Dataset1(root=cfg.TRAIN.DATA_ROOT, split='train', transform=transform_train)
    val_dataset = Dataset1(root=cfg.TRAIN.DATA_ROOT, split='val', transform=transform_val)

    print(f"Train dataset length: {len(train_dataset)}, Validation dataset length: {len(val_dataset)}")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=val_sampler, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    # Select model based on configuration
    if cfg.MODEL.TYPE == 'fasterrcnn':
        model = fasterrcnn_model(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.TYPE == 'fcos':
        model = fcos_model(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.TYPE == 'retinanet':
        model = retinanet_model(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unsupported model type: {cfg.MODEL.TYPE}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.TRAIN.BASE_LR, steps_per_epoch=len(train_loader), pct_start=cfg.TRAIN.PCT_START, anneal_strategy=cfg.TRAIN.ANNEAL_STRATEGY, epochs=cfg.TRAIN.EPOCHS)
    # scheduler = WarmupLR(scheduler, cfg.TRAIN.WARMUP_LR, cfg.TRAIN.WARMUP_EPOCHS, warmup_strategy='cos')

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, accuracies = train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, cfg.TRAIN.EPOCHS, device, scheduler)

    write_results_to_csv(cfg.TRAIN.RESULTS_CSV + "/" + cfg.TRAIN.RUN_NAME, train_losses)

    # Save results and model only if this is the primary process
    if args.rank == 0:  # Only save on the main process
        state_dict = model.module.state_dict() if args.distributed else model.state_dict()
        if cfg.TRAIN.SAVE_MODEL_PATH:
            save_model(state_dict, cfg.TRAIN.SAVE_MODEL_PATH + "/" + cfg.TRAIN.RUN_NAME)
            
    if args.distributed:
        cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Faster R-CNN model.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    parser.add_argument('--opts', nargs='*', help='Modify config options using the command line')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank of the current process')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')

    args = parser.parse_args()

    if args.config:
        cfg = CN.load_cfg(open(args.config, 'r'))
    else:
        cfg = get_cfg_defaults()

    if args.opts:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            key_list = k.split('.')
            d = cfg
            for key in key_list[:-1]:
                d = d[key]
            d[key_list[-1]] = eval(v)

    main(cfg, args)
