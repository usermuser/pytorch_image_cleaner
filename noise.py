import os
import cv2
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# from google.colab.patches import cv2_imshow
# from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.rcsetup as rcsetup

from model import MyModel, AutoencoderModel

# print(rcsetup.all_backends)

# plt.switch_backend('svg')

# img = os.listdir('content/train')[0]
# fix, ax = plt.subplots(1,2, figsize=(20, 10))
# img_noisy = mpimg.imread(f"content/train/{img}")
# img_clean = mpimg.imread(f"content/train_cleaned/{img}")
#
# cur_backend = plt.get_backend()
# print(cur_backend)

# ax[0].imshow(img_noisy, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Noisy', fontsize=20)

# ax[1].imshow(img_clean, cmap="gray")
# ax[1].axis("off")
# ax[1].set_title("Clean", fontsize=20)

# concatenate image Horizontally
# Hori = np.concatenate((img_clean, img_noisy), axis=1)
# cv2.imshow('graycsale image', Hori)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Сформирую выборку для тренировки модели и для проверки её качества:
train_imgs, test_imgs = train_test_split(
    os.listdir('content/train'),
    test_size=0.33,
    random_state=123,
)


class NoisyCleanDataset(Dataset):
    """
    Для удобства работы с данными создам вспомогательный класс на основе torch.utils.data.Dataset.
    Переопределю методы __len__ и __getitem__ под наши данные.

    Данный класс будет принимать на вход путь к папке с зашумленными изображениями,
    сам список с изображениями и, еще по необходимости,
    путь до чистых файлов и объект класса torchvision.transforms.transforms,
    содержащий список трансформаций над изображениями.
    При обращении по индексу данный класс будет возвращать кортеж
    из зашумленного изображения в формате torch.Tensor и его название в каталоге.
    Если указана директория для чистых изображений, то будет возвращено еще изображение без шума.


    """
    def __init__(self, noisy_path, images, clean_path=None, transforms=None):
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """
    При обращении по индексу данный класс будет возвращать кортеж
    из зашумленного изображения в формате torch.Tensor и его название в каталоге.
    Если указана директория для чистых изображений, то будет возвращено еще изображение без шума.
        """
        noisy_image = cv2.imread(f"{self.noisy_path}/{self.images[i]}")
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

        if self.transforms:
            noisy_image = self.transforms(noisy_image)

        if self.clean_path is not None:
            clean_image = cv2.imread(f"{self.clean_path}/{self.images[i]}")
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
            clean_image = self.transforms(clean_image)
            return (noisy_image, clean_image, self.images[i])
        else:
            return noisy_image, self.images[i]


def show_pair_img(img):
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    img_noisy = cv2.resize(mpimg.imread(f'content/train/{img}'), (400, 400))
    img_clean = cv2.resize(mpimg.imread(f'content/train_cleaned/{img}'), (400, 400))
    img_cleaned = mpimg.imread(f'outputs/{img}')

    ax[0].imshow(img_clean, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Clean', fontsize=20)

    ax[1].imshow(img_noisy, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Noisy', fontsize=20)

    ax[2].imshow(img_cleaned, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Cleaned', fontsize=20)
    plt.show()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Определю трансформации для изображения в переменной transform.
# При трансформации изображение будет изменено под размер 400 на 400 пикселей
# и возвращено в формате torch.Tensor.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

if __name__ == '__main__':

    AutoEncoder = MyModel(NoisyCleanDataset, AutoencoderModel, transform)
    AutoEncoder.show_info()
    train_path = 'content/train'
    trained_path = 'content/train_cleaned'
    # AutoEncoder.fit(
    #     40,
    #     train_path,
    #     trained_path,
    #     train_imgs,
    #     test_imgs,
    # )

    # Сохраню веса обученной модели:
    # AutoEncoder.save_weights('model1.pth')
    AutoEncoder.load_weights('model1.pth')

    # Визуализирую процесс обучения:
    # f, ax = plt.subplots(figsize=(10, 10))
    # ax.plot(AutoEncoder.train_loss, color='red', label='train')
    # ax.plot(AutoEncoder.val_loss, color='green', label='val')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.legend()
    # plt.show()

    AutoEncoder.predict('content/train')
    for i in range(3):
        show_pair_img(test_imgs[i])
