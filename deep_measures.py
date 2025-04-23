import torch
import lpips
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

# Initialize LPIPS model (downloads weights on first run)
_lpips_model = lpips.LPIPS(net='vgg').eval()

# Initialize a pretrained VGG16 and cut off after conv3_3 for feature extraction
_vgg = models.vgg16(pretrained=True).features[:16].eval()
for p in _vgg.parameters():
    p.requires_grad = False

# Common transform: HWC uint8 → CHW float tensor normalized for both LPIPS and VGG
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),              # VGG expects ≥224×224
    T.ToTensor(),                      # → [0,1]
    T.Normalize(                       # Imagenet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def to_fake_rgb(frame: np.ndarray) -> np.ndarray:
    """
    IF NEEDED! 
    Expand a single-channel image to 3 channels by repeating the gray values.
    """
    if frame.ndim == 2:
        # (H, W) → (H, W, 1)
        frame = frame[:, :, None]
    if frame.ndim == 3 and frame.shape[2] == 1:
        # repeat the single channel 3 times
        return np.repeat(frame, 3, axis=2)
    else:
        return frame

def perceptual(frame1: np.ndarray,
                  frame2: np.ndarray,
                  device: str = 'cpu') -> float:
    """
    Compute the LPIPS perceptual distance between two frames.
    LPIPS stands for Learned Perceptual Image Patch Similarity.
    """
    frame1 = to_fake_rgb(frame1)
    frame2 = to_fake_rgb(frame2)
    
    # Prepare tensors in [-1,1] as LPIPS expects
    def to_lpips_tensor(f):
        t = T.ToTensor()(f) * 2 - 1    # [0,1]→[-1,1]
        return t.unsqueeze(0).to(device)
    
    img1 = to_lpips_tensor(frame1)
    img2 = to_lpips_tensor(frame2)
    _lpips_model.to(device)
    with torch.no_grad():
        dist = _lpips_model(img1, img2)
    return float(dist.item())

def deep_cosine(frame1: np.ndarray,
                                frame2: np.ndarray,
                                device: str = 'cpu') -> float:
    """
    Compute cosine similarity between deep features (VGG16 conv3_3) of two frames.
    """
    frame1 = to_fake_rgb(frame1)
    frame2 = to_fake_rgb(frame2)
    
    # Transform and move to device
    t1 = _transform(frame1).unsqueeze(0).to(device)
    t2 = _transform(frame2).unsqueeze(0).to(device)
    _vgg.to(device)
    with torch.no_grad():
        f1 = _vgg(t1)  # shape [1, C, H', W']
        f2 = _vgg(t2)
    v1 = f1.view(1, -1)
    v2 = f2.view(1, -1)
    cos_sim = F.cosine_similarity(v1, v2, dim=1)
    # converted to distance for consistency
    return 1 - float(cos_sim.item())
