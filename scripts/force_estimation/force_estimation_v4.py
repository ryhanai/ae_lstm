import torch
import torch.nn as nn
from torchinfo import summary
import cv2
import torchvision.transforms as T


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
    def __init__(self, device=0, fine_tune_encoder=True):
        super().__init__()

        self.stdev = 0.02
        self.device = device

        self.augmenter = T.Compose([
            # T.ToTensor(),
            T.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03)),
            T.ColorJitter(hue=0.1, saturation=0.1),
            T.RandomAutocontrast(),
            T.ColorJitter(contrast=0.1, brightness=0.1),
        ])

        # image -> torch.Size([1, 384])
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if not fine_tune_encoder:
            for p in self.encoder.parameters():
                p.require_drad = False

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
            nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.augmenter(x) + torch.normal(mean=0, std=self.stdev, size=x.shape).to(self.device)
        return self.decoder(self.encoder(x))


class ForceEstimationResNet(nn.Module):
    def __init__(self, fine_tune_encoder=True):
        super().__init__()

    def forward(self, x):
        return self.decoder(self.encoder(self.augmenter(x)))

# print(summary(self, input_size=(32, 3, 224, 224)))
# model = ForceEstimationDINOv2()

