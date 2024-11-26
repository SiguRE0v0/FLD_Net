import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from Utils import preprocess, traversal

class FPDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_size=224):
        self.img_dir = img_dir
        self.img = []
        self.label = []
        self.transform = transform
        self.img_size = img_size

        logging.info('Creating dataset')
        self.img, self.label = traversal.file_traversal(img_dir)
        logging.info(f'Finished creating dataset with {len(self.img)} images')

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_path = self.img[idx]
        label = self.label[idx]
        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        image = preprocess.patch(img, self.img_size)
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        return image.float().contiguous(), label