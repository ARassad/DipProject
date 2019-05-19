import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, input_dir, img_res=(128, 128)):
        self.input_dir = input_dir
        self.img_res = img_res
        self.n_batches = -1

    def load_data(self, batch_size=1, is_testing=False):
        source_path = glob('{}{}source/*'.format(self.input_dir, "" if self.input_dir[-1] == "\\" else "\\"))
        pose_path = glob('{}{}pose/*'.format(self.input_dir, "" if self.input_dir[-1] == "\\" else "\\"))

        batch_images_idx = np.random.choice(range(len(source_path)), size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_idx in batch_images_idx:
            img_A = scipy.misc.imresize(self.imread(source_path[img_idx]), self.img_res)
            img_B = scipy.misc.imresize(self.imread(pose_path[img_idx]), self.img_res)

            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        source_path = glob('{}{}source/*'.format(self.input_dir, "" if self.input_dir[-1] == "\\" else "\\"))
        pose_path = glob('{}{}pose/*'.format(self.input_dir, "" if self.input_dir[-1] == "\\" else "\\"))

        self.n_batches = int(len(source_path) / batch_size)
        for i in range(self.n_batches-1):
            source_batch_path = source_path[i * batch_size:(i + 1) * batch_size]
            pose_batch_path = pose_path[i * batch_size:(i + 1) * batch_size]

            imgs_A, imgs_B = [], []
            for i in range(batch_size):
                img_A = scipy.misc.imresize(self.imread(source_batch_path[i]), self.img_res)
                img_B = scipy.misc.imresize(self.imread(pose_batch_path[i]), self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
