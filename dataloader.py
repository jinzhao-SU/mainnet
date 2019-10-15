from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys


class UAVDatasetTuple(Dataset):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.label_md = []
        self.image_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        self.image_md = np.load(self.image_path).astype(float)
        self.label_md = np.load(self.label_path).astype(float)
        assert len(self.image_md) == len(self.label_md), "not identical"

    def __getitem__(self, idx):
        sample = {}
        try:
            image = self._prepare_image(idx)
            label = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        sample = {'image': image, 'label': label}

        return sample

    def _prepare_image(self, idx):
        image_md = self.image_md[idx]
        return image_md

    def _get_label(self, idx):
        label_md = self.label_md[idx]
        return label_md

    def get_class_count(self):
        total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
        positive_class = 0
        for label in self.label_md:
            positive_class += np.sum(label)
        print("The number of positive image pair is:", positive_class)
        print("The number of negative image pair is:", total - positive_class)
        positive_ratio = positive_class / total
        negative_ratio = (total - positive_class) / total

        return positive_ratio, negative_ratio
