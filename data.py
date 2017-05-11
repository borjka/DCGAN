import pickle
import numpy as np
import math
from PIL import Image
import time


class Cifar10Data:
    names = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
        }

    data = None
    labels = None

    def __init__(self):
        self._images = []
        self._class_name = None
        self._images_numb = 0

    @classmethod
    def load_raw_data(cls, dir_name='cifar-10-batches-py/'):
        if cls.data is None:
            for i in range(1, 6):
                file_name = dir_name + str(i)
                with open(file_name, 'rb') as fo:
                    dict_with_data = pickle.load(fo, encoding='bytes')
                    if cls.data is None:
                        cls.data = dict_with_data[b'data']
                        cls.labels = dict_with_data[b'labels']
                    else:
                        cls.data = np.vstack((cls.data, dict_with_data[b'data']))
                        cls.labels += dict_with_data[b'labels']
        else:
            print('Data already loaded')

    @staticmethod
    def cifar_to_np(arr, N_pxls=1024):
        h = w = int(math.sqrt(N_pxls))
        img = np.zeros([h, w, 3], dtype='uint8')
        for i in range(N_pxls):
            pxl = (arr[i], arr[i + N_pxls], arr[i + N_pxls * 2])
            img[i // h, i % w, :] = pxl
        return img

    def set_images_of_class(self, class_name='cat'):
        if self.data is None:
            print('\nATTENTION!')
            print('Data hasn\'t been loaded.')
            print('Just add string \"CifarData.load_raw_data()\" in the beginning of your code.')
        else:
            if self._images:
                self._images = []
                self._images_numb = 0
            self._class_name = class_name
            for i, label in enumerate(self.labels):
                # DEBUG PARAMETER. REMOVE THEN
                if i == 100:
                    break
                ##############################
                if label == self.names[class_name]:
                    self._images.append(self.cifar_to_np(self.data[i]))
            self._images_numb = len(self._images)

    def show_img(self, index=0, original=False, grayscale=False):
        img = Image.fromarray(self._images[index], 'RGB')
        if grayscale:
            img = img.convert('L')
        if not original:
            img.resize((100, 100), Image.ANTIALIAS).show()
        else:
            img.show()

    def show_all_images(self):
        for i in range(self._images_numb):
            self.show_img(index=i)

    def print_sys_info(self):
        if self._class_name is None:
            print('Images haven\'t been loaded')
        else:
            shape = self._images[0].shape
            print('\n' + self._class_name.upper() + 'S')
            print('Number of images: {0}'.format(self._images_numb))
            print('Image size: {0}x{1}'.format(shape[0], shape[1]))


def main():
    start_time = time.time()

    data = Cifar10Data()
    Cifar10Data.load_raw_data()
    data.set_images_of_class(class_name='dog')
    data.show_all_images()
    data.print_sys_info()

    print('Execution time: ' + str(time.time() - start_time))


if __name__ == '__main__':
    main()




