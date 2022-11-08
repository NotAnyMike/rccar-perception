# Create a dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import PIL


class YoloFormatDataset(Dataset):
    def __init__(self, images, labels, input_shape):
        self.data = data
        self.labels = labels
        self.input_shape = input_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image from the path in data[idx]
        image = PIL.Image.open(self.data[idx])
        img = np.zeros(input_shape + (3, ))

        scale = min(input_shape[0] / image.size[0], input_shape[1] / image.size[1])
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))

        # Tradeoff can be found here 
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
        image = image.resize(new_size, PIL.Image.BICUBIC)
        img[:new_size[1], :new_size[0], :] = np.array(image)

        # Read vals from labels[idx] in yolo format
        with open(self.labels[idx], 'r') as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                line = line.split()
                labels.append([float(x) for x in line])
        labels = np.array(labels)

        return img, labels


class YoloDataModule(pl.LightningDataModule):
    def __init__(self, folder, input_shape):
        """Folder contains two subfolders: images and labels"""
        super().__init__()

        self.data = []
        self.labels = []
        self.input_shape = input_shape

        for img_path in folder.glob('images/*'):
            label_path = folder / 'labels' / img_path.stem + '.txt'
            self.data.append(img_path)
            self.labels.append(label_path)

    def setup(self, split:.8):
        """Split data into train and val randomly"""
        n = len(self.data)
        n_train = int(n * split)
        idx = torch.randperm(n)
        self.train_data = self.data[idx[:n_train]]
        self.train_labels = self.labels[idx[:n_train]]
        self.val_data = self.data[idx[n_train:]]
        self.val_labels = self.labels[idx[n_train:]]

    def train_dataloader(self):
        return DataLoader(YoloFormatDataset(self.train_data,
                                            self.train_labels,
                                            self.input_shape),
                          batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(YoloFormatDataset(self.val_data,
                                            self.val_labels,
                                            self.input_shape),
                          batch_size=32, shuffle=False, num_workers=4)
