"""
Step 0c: Architecture sanity check for MK-UNet.
No dataset required. Just confirms the model builds and matches
the paper's claimed params/FLOPs before any training.

Usage:
    python sanity_check.py
"""
import torch
from thop import profile
from mkunet_network import MK_UNet

NET_CONFIGS = {
    'MK_UNet_T': [4, 8, 16, 24, 32],
    'MK_UNet_S': [8, 16, 32, 48, 80],
    'MK_UNet':   [16, 32, 64, 96, 160],
    'MK_UNet_M': [32, 64, 128, 192, 320],
    'MK_UNet_L': [64, 128, 256, 384, 512],
}

# Paper's claimed numbers (Table 1, 256x256 inputs) for reference
PAPER_CLAIMS_256 = {
    'MK_UNet_T': (0.027, 0.062),
    'MK_UNet_S': (0.093, 0.125),
    'MK_UNet':   (0.316, 0.314),
    'MK_UNet_M': (1.15, 0.951),
    'MK_UNet_L': (3.76, 3.19),
}

def check(name, channels, img_size):
    model = MK_UNet(num_classes=1, in_channels=3, channels=channels).cuda().eval()
    dummy = torch.randn(1, 3, img_size, img_size).cuda()
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- {name} @ {img_size}x{img_size} ---")
    print(f"  thop params: {params/1e6:.4f}M   thop FLOPs: {flops/1e9:.4f}G")
    print(f"  raw total params: {total_params/1e6:.4f}M")

    if img_size == 256 and name in PAPER_CLAIMS_256:
        p_claim, f_claim = PAPER_CLAIMS_256[name]
        print(f"  paper claims (256x256): {p_claim}M params, {f_claim}G FLOPs")
        print(f"  params match: {abs(params/1e6 - p_claim) < 0.01}")

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA not available -- fix environment first."
    print(f"Torch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

    # Check all variants at 256x256 (matches paper's Table 1 FLOPs convention)
    for name, channels in NET_CONFIGS.items():
        check(name, channels, img_size=256)

    check('MK_UNet', NET_CONFIGS['MK_UNet'], img_size=352)
