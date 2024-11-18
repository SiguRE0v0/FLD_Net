import torch
from tqdm import tqdm

@torch.inference_mode()
def validation(model, val_loader, device):
    model.eval()
    num_val_batches = len(val_loader)
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=num_val_batches, desc='Validation round', unit='img', position=0, leave=False) as pbar:
            for images, labels in val_loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                outputs, _ = model(images)

                _, pred = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (pred == labels).sum().item()
                pbar.update(images.shape[0])

    model.train()
    accuracy = correct / total
    return accuracy