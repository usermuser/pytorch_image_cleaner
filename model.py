import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Создам модель. Она будет состоять из двух блоков.
# Первый блок (энкодер) уменьшает размерность изображения, извлекая из него нужные признаки.
# Второй блок (декодер) пытается восстановить изображения по извлеченным признакам
class AutoencoderModel(nn.Module):
    def __init__(self):
        super(AutoencoderModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = nn.functional.interpolate(encoded, scale_factor=2)
        decoded = self.decoder(decoded)
        return decoded


# Создам класс, в котором реализуем обучение и предсказание модели.
# А также добавлю возможность выводить информацию о модели, сохранять и загружать веса модели.
class MyModel():
    def __init__(self, Dataset, Model, transforms):
        self.Dataset = Dataset
        self.model = Model().to(device)
        self.transform = transforms

    def load_weights(self, path):
        if device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def show_info(self):
        print(summary(self.model, (1, 400, 400)))

    def fit(self, n_epochs, noisy_path, clean_path, train_imgs, test_imgs):

        train_data = self.Dataset(noisy_path, train_imgs, clean_path, self.transform)
        val_data = self.Dataset(noisy_path, test_imgs, clean_path, self.transform)

        trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
        valloader = DataLoader(val_data, batch_size=4, shuffle=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            verbose=True
        )

        self.model.train()
        self.train_loss = []
        self.val_loss = []
        running_loss = 0.0

        for epoch in range(n_epochs):
            self.model.train()
            for i, data in enumerate(trainloader):
                noisy_img = data[0]
                clean_img = data[1]
                noisy_img = noisy_img.to(device)
                clean_img = clean_img.to(device)
                optimizer.zero_grad()
                outputs = self.model(noisy_img)
                loss = criterion(outputs, clean_img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    print(f'Epoch {epoch + 1} batch {i}: Loss {loss.item() / 4}')
            self.train_loss.append(running_loss / len(trainloader.dataset))
            print('Validation ...')
            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(enumerate(valloader),
                                    total=int(len(val_data) / valloader.batch_size)):
                    noisy_img = data[0]
                    clean_img = data[1]
                    noisy_img = noisy_img.to(device)
                    clean_img = clean_img.to(device)
                    outputs = self.model(noisy_img)
                    loss = criterion(outputs, clean_img)
                    running_loss += loss.item()
                current_val_loss = running_loss / len(valloader.dataset)
                self.val_loss.append(current_val_loss)
                print(f"Val Loss: {current_val_loss:.5f}")

    def predict(self, img):
        os.makedirs('outputs', exist_ok=True)
        self.model.eval()
        if type(img) == str:
            if os.path.isfile(img):
                filename = os.path.basename(img)
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = self.transform(img).to(device)
                img = self.model(img)
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(f'outputs/{filename}', img * 255)
            else:
                images = os.listdir(img)
                predictDataset = self.Dataset(img, images, transforms=self.transform)
                predictDataloader = DataLoader(predictDataset, batch_size=4, shuffle=False)
                with torch.no_grad():
                    for i, data in tqdm(enumerate(predictDataloader), total=int(
                            len(predictDataset) / predictDataloader.batch_size)):
                        noisy_img = data[0]
                        noisy_img = noisy_img.to(device)
                        outputs = self.model(noisy_img)
                        for im, image_name in zip(outputs, data[1]):
                            im = im.detach().cpu().permute(1, 2, 0).numpy()
                            cv2.imwrite(f'outputs/{image_name}', im * 255)
        if type(img) == np.ndarray:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.transform(img).to(device)
            img = self.model(img)
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            cv2.imwrite('outputs/cleaned_img.jpg', img * 255)
