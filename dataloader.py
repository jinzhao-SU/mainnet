from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys


class UAVDatasetTuple(Dataset):
    def __init__(self, image_path, init_path, label_path):
        self.image_path = image_path
        self.init_path = init_path
        self.label_path = label_path
        self.label_md = []
        self.image_md = []
        self.init_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        # print("gettuple")
        self.image_md = np.load(self.image_path).astype(float)
        # print("test", self.image_md.shape)
        self.image_md = np.moveaxis(self.image_md, 1, -1)
        self.init_md = np.load(self.init_path).astype(float)
        self.label_md = np.load(self.label_path).astype(float)
        assert len(self.image_md) == len(self.label_md), "not identical"

    def __getitem__(self, idx):
        sample = {}
        try:
            # print("getitem")
            image, init = self._prepare_image(idx)
            label = self._get_label(idx)

            image = np.expand_dims(image, axis=0)
            init = np.expand_dims(init, axis=0)

        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        sample = {'image': image, 'init': init, 'label': label}

        return sample

    def _prepare_image(self, idx):
        image_md = self.image_md[idx]
        init_md = self.init_md[idx]
        return image_md, init_md

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

if __name__ == '__main__':
    data_path ='/data/zzhao/uav_regression/main_test/data_subnet_output.npy'
    init_path = '/data/zzhao/uav_regression/main_test/data_init_density.npy'
    label_path = '/data/zzhao/uav_regression/main_test/label_T1_10s.npy'

    all_dataset = UAVDatasetTuple(image_path=data_path, init_path=init_path, label_path=label_path)
    sample = all_dataset[0]
    print(sample['image'].shape)

# init 100*100
# suboutput 60*100*100
# label 100*100