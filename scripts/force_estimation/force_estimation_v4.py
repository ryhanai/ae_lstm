import torch
import torch.nn as nn
from torchinfo import summary
import cv2

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# img = cv2.imread('/home/ryo/Dataset/dataset2/pen-kitting-real/1/image_frame00000.jpg')
# img2 = torch.tensor(cv2.resize(img, (224,224)) / 255, dtype=torch.float32)
# img3 = torch.permute(img2, [2, 0, 1])
# dinov2_vits14(img3.unsqueeze(0))


class ForceEstimationDINOv2(nn.Module):
    """
        input: (3, 336, 672): width and height must be a multple of 14
        output: (40, 120, 160)
    """
    def __init__(self):
        super().__init__()

        # image -> torch.Size([1, 384])
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # self.decoder = nn.Sequential(
        #     nn.Linear(384, 50), nn.BatchNorm1d(50), nn.ReLU(True),
        #     nn.Linear(50, 30*40*8), nn.BatchNorm1d(30*40*8), nn.ReLU(True),
        #     nn.Unflatten(1,(30,40,8)),
        #     nn.ConvTranspose2d(8, 16, 3, 2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 30, 3, 2, padding=1), nn.Tanh(),
        # )

        self.decoder = nn.Sequential(
            nn.Linear(384, 50), nn.BatchNorm1d(50), nn.ReLU(True),
            nn.Linear(50, 8*30*40), nn.BatchNorm1d(8*30*40), nn.ReLU(True),
            nn.Unflatten(1, (8, 30, 40)),
            nn.ConvTranspose2d(8, 16, 3, 2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# model = ForceEstimationDINOv2()
# print(summary(model, input_size=(32, 3, 224, 224)))
