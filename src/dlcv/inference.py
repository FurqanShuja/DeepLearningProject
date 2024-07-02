import argparse
import torch
import os
import json
from tqdm import tqdm
from torchvision.ops import nms

from dlcv.config import get_cfg_defaults, CN
from dlcv.utils import custom_collate_fn, get_transforms, json_serializable
from dlcv.dataset2 import Dataset2
from dlcv.model import fasterrcnn_model, fcos_model, retinanet_model
from torch.utils.data import DataLoader

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

def evaluate_one_epoch(model, data_loader, device, save_dir, score_threshold, iou_threshold):
    model.eval()
    results = []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, filenames = batch
            images = [image.to(device) for image in images]

            outputs = model(images)
            if not outputs:
                print("No outputs from model for these images.")
                continue

            for i, output in enumerate(outputs):
                file_name = filenames[i]
                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']

                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep].cpu().numpy()
                scores = scores[keep].cpu().numpy()
                labels = labels[keep].cpu().numpy()

                score_filter = scores >= score_threshold
                boxes = boxes[score_filter]
                scores = scores[score_filter]
                labels = labels[score_filter]

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width, height = x_max - x_min, y_max - y_min
                    result = {
                        'file_name': file_name,
                        'category_id': int(label),
                        'bbox': [float(x_min), float(y_min), float(width), float(height)],
                        'score': float(score)
                    }
                    results.append(result)

    results = json_serializable(results)

    results_json_path = os.path.join(save_dir, "predictions.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_json_path}")

def main(cfg, args):
    if args.distributed:
        init_distributed_mode(args)

    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.TRAIN.NO_CUDA else 'cpu')

    transform_test = get_transforms(train=False, 
                                    image_height=cfg.TEST.IMAGE_HEIGHT, 
                                    image_width=cfg.TEST.IMAGE_WIDTH)

    test_dataset = Dataset2(root=cfg.TRAIN.DATA_ROOT, split='test', transform=transform_test)

    print(f"Test dataset length: {len(test_dataset)}")

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank)
    else:
        test_sampler = torch.utils.data.RandomSampler(test_dataset)
        
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, sampler=test_sampler, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    model_path = os.path.join(cfg.TRAIN.SAVE_MODEL_PATH, cfg.TRAIN.RUN_NAME + ".pth")
    # Select model based on configuration
    if cfg.MODEL.TYPE == 'fasterrcnn':
        model = fasterrcnn_model(num_classes=cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.TYPE == 'fcos':
        model = fcos_model(num_classes=cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.TYPE == 'retinanet':
        model = retinanet_model(num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model type: {cfg.MODEL.TYPE}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    evaluate_one_epoch(model, test_loader, device, cfg.TRAIN.RESULTS_CSV, cfg.TEST.SCORE_THRESHOLD, cfg.TEST.IOU_THRESHOLD)

    if args.distributed:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model.')
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
