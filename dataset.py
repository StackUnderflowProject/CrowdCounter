import random
from torch.utils.data import Dataset

from utils import load_data

class ListDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        self.shuffle = shuffle
        if train:
            root = root * 4
        if shuffle:
            random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)

        if self.transform is not None:
            img = self.transform(img)
        return img, target