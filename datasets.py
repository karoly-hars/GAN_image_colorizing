import os
import cv2
import pickle
import numpy as np
from torch.utils.data import Dataset
import random


def unpickle_batch(file):
    with open(file, 'rb') as f:
        dict_ = pickle.load(f, encoding='bytes')
    return dict_


def preprocess(img_bgr):
    # to 32bit img
    img_bgr = img_bgr.astype(np.float32)/255.0
    # transform to lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # normalize
    img_lab[:, :, 0] = img_lab[:, :, 0]/50 - 1
    img_lab[:, :, 1] = img_lab[:, :, 1]/127
    img_lab[:, :, 2] = img_lab[:, :, 2]/127
    # transpose
    img_lab = img_lab.transpose((2, 0, 1))
    return img_lab


def postprocess(img_lab):
    # transpose back
    img_lab = img_lab.transpose((1, 2, 0))
    # transform back
    img_lab[:, :, 0] = (img_lab[:, :, 0] + 1)*50
    img_lab[:, :, 1] = img_lab[:, :, 1]*127
    img_lab[:, :, 2] = img_lab[:, :, 2]*127
    # transform to bgr
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # to int8
    img_bgr = (img_bgr*255.0).astype(np.uint8)
    return img_bgr


class Cifar10Dataset(Dataset):
    def __init__(self, root_dir, mirror=False, random_seed=None):
        self.img_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        if random_seed is not None:
            self.img_paths.sort()
            random.Random(random_seed).shuffle(self.img_paths)
        self.mirror = mirror

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_bgr = cv2.imread(img_path)

        if self.mirror:
            if random.random() > 0.5:
                img_bgr = img_bgr[:, ::-1, :]

        img_lab = preprocess(img_bgr)
        return img_lab

    @classmethod
    def get_datasets_from_scratch(cls, data_path):
        """Download and extract dataset + create dataset objects."""
        cls.get_cifar10_data(data_path)
        data_dirs = cls.extract_cifar10_images(data_path)

        datasets = dict()
        datasets['train'] = cls(root_dir=data_dirs['train'], mirror=True)
        datasets['test'] = cls(root_dir=data_dirs['test'], mirror=False, random_seed=1)

        return datasets

    @staticmethod
    def get_cifar10_data(data_path):
        """Download and uncompress CIFAR10 dataset."""
        if not os.path.exists(data_path):
            # download
            print('Downloading dataset...')
            import urllib.request
            os.makedirs(data_path)
            urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                                       os.path.join(data_path, 'cifar-10-python.tar.gz'))

            # extract file
            print('unzipping dataset...')
            import tarfile
            tar = tarfile.open(os.path.join(data_path, 'cifar-10-python.tar.gz'), 'r:gz')
            tar.extractall(path=data_path)
            tar.close()
        else:
            print('cifar10 already downloaded.')

    @staticmethod
    def extract_cifar10_images(data_path):
        """Restructure the CIFAR10 images and split it into train and test."""
        # extract images from batches
        data_batches = dict()
        data_batches['test'] = [os.path.join(data_path, 'cifar-10-batches-py', 'test_batch')]
        data_batches['train'] = [os.path.join(data_path, 'cifar-10-batches-py', f) for f in [
            'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'
        ]]

        data_dirs = dict()
        data_dirs['test'] = os.path.join(data_path, 'cifar-10-images', 'test')
        data_dirs['train'] = os.path.join(data_path, 'cifar-10-images', 'train')

        for phase in ['test', 'train']:
            if not os.path.exists(data_dirs[phase]):
                print('extracting {} images...'.format(phase))
                os.makedirs(data_dirs[phase])

                for data_batch in data_batches[phase]:
                    batch = unpickle_batch(data_batch)

                    for image_name, image_parts in zip(batch[b'filenames'], batch[b'data']):
                        r, g, b = image_parts[0:1024], image_parts[1024:2048], image_parts[2048:]
                        r, g, b = np.reshape(r, (32, -1)), np.reshape(g, (32, -1)), np.reshape(b, (32, -1))
                        img = np.stack((b, g, r), axis=2)

                        save_path = os.path.join(data_dirs[phase], image_name.decode('utf-8'))
                        cv2.imwrite(save_path, img)
            else:
                print('{} image set already extracted'.format(phase))

        return data_dirs
