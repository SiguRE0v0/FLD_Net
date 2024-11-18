import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import os
from model import FPLDNet
from Utils.dataset import FPDataset
from Utils.evaluate import validation
from Utils.preprocess import RandomRotate90Degree

#os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-m', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--size', '-s', type=int, default=224, help='Size of the images after preprocess', dest='size')
    parser.add_argument('--lambda', '-w', type=float, default=0.2, help="The weight of the Auxiliary Classifier's loss",dest='factor')
    parser.add_argument('--numval', '-n', type=int, default=2, help="The number of validation round in each epoch", dest='num_val')
    return parser.parse_args()

#dir_img = '/home/share/dataset/training'
dir_img = './data/training/'
dir_checkpoint = './checkpoints'

def train_model(
        model,
        device,
        epochs,
        batch_size,
        learning_rate,
        val_percent,
        amp,
        img_size,
        factor,
        save_checkpoint: bool = True
):
    # Create Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomRotation(90),
        RandomRotate90Degree()
    ])
    dataset = FPDataset(dir_img, img_size = img_size)

    # Split into train / validation set
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set.transform = transform
    val_set.transform = None

    # Create dataloader
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, batch_size=1)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
            Images size:     {img_size}
            loss lambda:     {factor}
        ''')

    # Set up the optimizer, the loss and the learning rate
    optimizer = optim.AdamW(model.parameters(), learning_rate)
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=6, factor = 0.5,
                                                     threshold = 0.01, min_lr = 1e-6)
    criterion = nn.CrossEntropyLoss()

    # Begin training
    global_step = 0
    best_acc = 0
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        final_accuracy = 0
        train_acc = 0
        total = 0
        correct = 0
        num_val = args.num_val
        division_step = n_train // num_val
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', position=0, leave=True, unit = 'img') as pbar:
            for batch in train_loader:
                images, labels = batch
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                #     main_logits, auxiliary_logits = model(images)
                #     main_loss = criterion(main_logits, labels)
                #     sub_loss = criterion(auxiliary_logits, labels)
                #     loss = main_loss + factor * sub_loss
                # Loss
                main_logits, auxiliary_logits = model(images)
                main_loss = criterion(main_logits, labels)
                sub_loss = criterion(auxiliary_logits, labels)
                loss = main_loss + factor * sub_loss

                # accuracy in training
                total += images.size(0)
                _, pred = torch.max(main_logits, 1)
                correct += torch.eq(pred, labels).sum().item()
                train_acc = correct / total

                #grad_scaler.scale(loss).backward()
                loss.backward()
                optimizer.step()
                global_step += 1
                #grad_scaler.unscale_(optimizer)
                #grad_scaler.step(optimizer)
                #grad_scaler.update()

                current_lr = optimizer.param_groups[0]['lr']
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': loss.item(), 'learning rate': current_lr, 'train accuracy': train_acc})

                # Evaluation round during epoch
                if total >= division_step or total == n_train:
                    division_step += division_step
                    acc = validation(model, val_loader, device)
                    scheduler.step(acc)
                    final_accuracy = acc
                    if best_acc == 0:
                        best_acc = acc
                    if acc > best_acc:
                        best_acc = acc
                    print('\nValidation accuracy: {}'.format(acc))
                    print('Best accuracy: {}'.format(best_acc))

        logging.info(f'Epoch: {epoch}, Best acc: {best_acc}, Validate acc: {final_accuracy} , Train acc: {train_acc}')

        # Epoch finished, save model
        if save_checkpoint:
            state_dict = model.state_dict()
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            torch.save(state_dict, os.path.join(dir_checkpoint,'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = FPLDNet(input_size = args.size, patch_size = 16, embed_dim = 32, num_heads = 4)
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model = model.to(device)

    torch.cuda.empty_cache()
    train_model(
        model = model,
        device =device,
        epochs = args.epochs,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        val_percent = args.val,
        amp = args.amp,
        img_size = args.size,
        factor = args.factor
        )

