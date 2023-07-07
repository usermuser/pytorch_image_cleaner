from model import MyModel, AutoencoderModel
from noise import NoisyCleanDataset, transform

AutoEncoder = MyModel(NoisyCleanDataset, AutoencoderModel, transform)
AutoEncoder.load_weights('model1.pth')

AutoEncoder.predict('content/payload/payload2.jpg')


