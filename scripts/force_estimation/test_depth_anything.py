import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


def load_image(scene=3, view=0):
    image_path = f'/home/ryo/Dataset/forcemap/tabletop_airec241008/rgb{scene:05d}_{view:05d}.jpg'
    return cv2.imread(image_path)


# def infer(scene=3, view=0):
#     raw_img = load_image(scene=scene, view=view)
#     depth = model.infer_image(raw_img) # HxW raw depth map in numpy
#     return raw_img, depth


def infer(image, input_size=518):
    x, (h, w) = model.image2tensor(image, input_size)
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    print(f'h={h}, w={w}, patch_h={patch_h}, patch_w={patch_w}')
    features = model.pretrained.get_intermediate_layers(x, model.intermediate_layer_idx[model.encoder], return_class_token=True)
    return features


# class ForceEstimation(nn.Module):
#     def __init__(self):