import torch
import torch.nn as nn
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
    def __init__(self, device=0, fine_tune_encoder=False):
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

        # image -> torch.Size([BatchSize, 384])
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if not fine_tune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Head1: 
        #   fine-tune: train/test=0.00237/0.00246
        # self.decoder = nn.Sequential(
        #     nn.Linear(384, 50), nn.BatchNorm1d(50), nn.ReLU(True),
        #     nn.Linear(50, 8*30*40), nn.BatchNorm1d(8*30*40), nn.ReLU(True),
        #     nn.Unflatten(1, (8, 30, 40)),
        #     nn.ConvTranspose2d(8, 16, 3, 2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Sigmoid(),
        # )

        # Head2:
        #  fine-tune: train/test=
        # self.decoder = nn.Sequential(
        #     nn.Linear(384, 8*30*40), nn.BatchNorm1d(8*30*40), nn.ReLU(True),
        #     nn.Unflatten(1, (8, 30, 40)),
        #     nn.ConvTranspose2d(8, 16, 3, 2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Sigmoid(),
        # )

        # Head3:
        #  fine-tune: train/test=
        # self.decoder = nn.Sequential(
        #     nn.Linear(384, 16*60*80), nn.BatchNorm1d(16*60*80), nn.ReLU(True),
        #     nn.Unflatten(1, (16, 60, 80)),
        #     nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Sigmoid(),
        # )

        # Head4: 
        #   fine-tune: train/test=
        self.decoder = nn.Sequential(
            nn.Linear(384, 384), nn.BatchNorm1d(384), nn.ReLU(True),
            nn.Linear(384, 384), nn.BatchNorm1d(384), nn.ReLU(True),
            nn.Linear(384, 8*30*40), nn.BatchNorm1d(8*30*40), nn.ReLU(True),
            nn.Unflatten(1, (8, 30, 40)),
            nn.ConvTranspose2d(8, 16, 3, 2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 30, 3, 2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        # x = self.augmenter(x) + torch.normal(mean=0, std=self.stdev, size=x.shape).to(self.device)
        return self.decoder(self.encoder(x))


from torchvision.models import resnet50, ResNet50_Weights


class UpSample(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.resize = T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)  # adding antialias=True option may result in autograd error

    def forward(self, x):
        return self.resize(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, strides):
        super().__init__()

        if len(num_filters) == 1:
            num_filters = [num_filters[0], num_filters[0]]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, num_filters[0], kernel_size, strides[0], padding='same')
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_filters[0])
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, strides[0], padding='same')
        self.bn3 = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.Conv2d(in_channels, num_filters[1], kernel_size, strides[0], padding='same')

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)

        x = self.conv3(x)
        x = self.bn3(x)

        x += x1
        return x


class ForceEstimationResNet(nn.Module):
    def __init__(self, device=0, fine_tune_encoder=True):
        super().__init__()

        self.stdev = 0.02
        self.device = device

        self.augmenter = T.Compose([
            # T.ToTensor(),
            T.Resize([360, 512], antialias=True),
            T.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03)),
            T.ColorJitter(hue=0.1, saturation=0.1),
            T.RandomAutocontrast(),
            T.ColorJitter(contrast=0.1, brightness=0.1),
        ])

        # resnet_classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
        # resnet_classifier = resnet50()  # no pretrained weight
        resnet_classifier = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.encoder = torch.nn.Sequential(*(list(resnet_classifier.children())[:-2]))

        if not fine_tune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            ResBlock(2048, [1024, 512], 3, strides=[1, 1]),
            UpSample([24, 32]),
            ResBlock(512, [256, 128], 3, strides=[1, 1]),
            UpSample([48, 64]),
            ResBlock(128, [64, 64], 3, strides=[1, 1]),
            UpSample([96, 128]),
            ResBlock(64, [32, 32], 3, strides=[1, 1]),
            nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            T.Resize([120, 160], antialias=True),
        )

    def forward(self, x):
        x = self.augmenter(x) + torch.normal(mean=0, std=self.stdev, size=[360,512]).to(self.device)
        return self.decoder(self.encoder(x))


class ForceEstimationDinoRes(nn.Module):
    def __init__(self, device=0, fine_tune_encoder=True):
        super().__init__()

        self.stdev = 0.02
        self.device = device

        self.augmenter = T.Compose([
            # T.ToTensor(),
            T.Resize([336, 672], antialias=True),
            T.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03)),
            T.ColorJitter(hue=0.1, saturation=0.1),
            T.RandomAutocontrast(),
            T.ColorJitter(contrast=0.1, brightness=0.1),
        ])

        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if not fine_tune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.bridge = nn.Sequential(
            nn.Linear(384, 256*12*16), nn.BatchNorm1d(256*12*16), nn.ReLU(True),
            nn.Unflatten(1, (256, 12, 16)),
        )

        # input: [256,12,16]
        self.decoder = nn.Sequential(
            ResBlock(256, [256, 256], 3, strides=[1, 1]),
            UpSample([24, 32]),
            ResBlock(256, [256, 128], 3, strides=[1, 1]),
            UpSample([48, 64]),
            ResBlock(128, [64, 64], 3, strides=[1, 1]),
            UpSample([96, 128]),
            ResBlock(64, [32, 32], 3, strides=[1, 1]),
            nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            T.Resize([120, 160], antialias=True),
        )

        # self.bridge = nn.Sequential(
        #     nn.Linear(384, 128*12*16), nn.BatchNorm1d(128*12*16), nn.ReLU(True),
        #     nn.Unflatten(1, (128, 12, 16)),
        #     nn.Conv2d(128, 2048, 1, stride=1, padding=0),  # 1x1 convolution
        #     nn.BatchNorm2d(2048), nn.ReLU(True),
        # )

        # # input: [2048,12,16]
        # self.decoder = nn.Sequential(
        #     ResBlock(2048, [1024, 512], 3, strides=[1, 1]),
        #     UpSample([24, 32]),
        #     ResBlock(512, [256, 128], 3, strides=[1, 1]),
        #     UpSample([48, 64]),
        #     ResBlock(128, [64, 64], 3, strides=[1, 1]),
        #     UpSample([96, 128]),
        #     ResBlock(64, [32, 32], 3, strides=[1, 1]),
        #     nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid(),
        #     T.Resize([120, 160], antialias=True),
        # )

    def forward(self, x):
        x = self.augmenter(x) + torch.normal(mean=0, std=self.stdev, size=[336,672]).to(self.device)
        return self.decoder(self.bridge(self.encoder(x)))


# from torchinfo import summary
# model = ForceEstimationDinoRes(fine_tune_encoder=False)
# print(summary(model, input_size=(32, 3, 336, 672)))

