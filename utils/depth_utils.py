# import torch
#
# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# for param in midas.parameters():
#     param.requires_grad = False
#
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.dpt_transform
# downsampling = 1
#
# def estimate_depth(img, mode='test'):
#     h, w = img.shape[1:3]
#     norm_img = (img[None] - 0.5) / 0.5
#     norm_img = torch.nn.functional.interpolate(
#         norm_img,
#         size=(384, 512),
#         mode="bicubic",
#         align_corners=False)
#
#     if mode == 'test':
#         with torch.no_grad():
#             prediction = midas(norm_img)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=(h//downsampling, w//downsampling),
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#     else:
#         prediction = midas(norm_img)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=(h//downsampling, w//downsampling),
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#     return prediction
#
import torchvision
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image

# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 构建预处理流程
transform = Compose([
    Resize((518, 518), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 加载 DepthAnythingV2 模型
def load_depth_model(mode='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[mode])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{mode}.pth', map_location='cpu'))
    model.to(device).eval()
    return model

# 推理深度图
def estimate_depth(img, model, mode='test'):
    if model is None:
        raise ValueError("Depth model is None. Make sure to pass it from train.py.")

    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    else:
        raise TypeError("Input image must be a torch.Tensor")

    img_np = (img_np * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 518, 518]

    with torch.no_grad():
        prediction = model(input_tensor)  # 通常应为 [1, 1, H_pred, W_pred]
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)  # -> [1, 1, H, W]

    # 插值到原图大小
    prediction = torch.nn.functional.interpolate(
        prediction,
        size=(img.shape[1], img.shape[2]),  # [H_orig, W_orig]
        mode="bicubic",
        align_corners=False
    ).squeeze()  # [H, W]

    return prediction

