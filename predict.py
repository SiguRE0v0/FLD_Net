import argparse
import logging
import os
import sys
import torch
from sympy import false
from torch.utils.data import DataLoader
from PIL import Image
from Utils.dataset import FPDataset
from model.FPLD import FLDNet
from tqdm import tqdm

dir_img = "./data/testing"

def get_args():
    parser = argparse.ArgumentParser(description='Predicting model')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--size', '-s', type=int, default=224, help='Size of the images after preprocess', dest='size')
    parser.add_argument('--load', '-m', type=str, default=None, help='Load .pth model')
    return parser.parse_args()

def predict(
        model,
        device,
        test_set
):

    model.eval()
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, pin_memory=True, num_workers=8)
    # Start predict
    correct = 0
    total = 0

    with tqdm(total=len(test_set), desc=f'Predicting', position=0, leave=False, unit='img') as pbar:
        for batch in test_loader:
            images, labels = batch
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                output, _ = model(images)
                _, pred = torch.max(output, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                pbar.update(images.shape[0])
        accuracy = correct / total
    model.train()
    return accuracy


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    test_set = FPDataset(dir_img, img_size=args.size, transform=None)

    model = FLDNet(input_size=args.size, patch_size=16, embed_dim=32, num_heads=4)
    model = model.to(device)

    if args.load is None:
        logging.error(f'No model loaded, check the path of .pth')
        sys.exit()

    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {args.load}')

    accuracy = predict(model=model, device=device, img_size=args.size, batch_size=args.batch_size, loader=test_loader)
    logging.info(f'Accuracy: {accuracy}')