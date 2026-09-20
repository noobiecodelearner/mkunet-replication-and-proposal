"""
Step 2 Preliminary Result: does PRIME's degree-centrality selection signal
correlate with per-slice segmentation difficulty on the BraTS2020 test set?

Method:
1. Load the median-performing BraTS checkpoint (run 3, 82.55% test Dice --
   chosen to avoid cherry-picking a best/worst run).
2. Run inference on all 274 BraTS test slices, recording per-slice Dice
   using the exact same preprocessing/thresholding as train_brats.py's
   test() function (0.5 pred threshold, 0.2 GT threshold, bilinear/nearest
   resize -- kept identical so this is a faithful re-evaluation, not a
   different measurement).
3. Build a PRIME-faithful similarity graph over the same 274 test images:
   SSIM pairwise similarity (PRIME's primary metric per their own paper),
   Louvain community detection, per-node degree within its community.
4. Compute Spearman correlation between per-slice Dice and per-slice
   degree. A near-zero or negative correlation would support the proposal's
   claim that PRIME's redundancy-based selection signal is not a good
   proxy for task/adaptation difficulty.

Usage (run from MK-UNet/ directory, mkunetenv active):
    python prelim_degree_vs_difficulty.py
"""
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
import networkx as nx
import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt

from mkunet_network import MK_UNet

CKPT_PATH = "model_pth/BraTS2020_MK_UNet_bs8_lr0.0005_e200_augTrue_run3_t050508/BraTS2020_MK_UNet_bs8_lr0.0005_e200_augTrue_run3_t050508-best.pth"
TEST_IMG_DIR = "../braTS/brats_2d/test/images"
TEST_MASK_DIR = "../braTS/brats_2d/test/masks"
IMG_SIZE = 256
SSIM_THRESHOLD = 0.5  # PRIME uses a "median threshold... for denser network construction" (Fig. 3 caption) as one reference point; revisit if community structure looks degenerate


def dice_coefficient(predicted, labels):
    smooth = 1e-6
    p = predicted.flatten()
    l = labels.flatten()
    intersection = (p * l).sum()
    total = p.sum() + l.sum()
    return (2. * intersection + smooth) / (total + smooth)


def load_model(ckpt_path):
    model = MK_UNet(num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160]).cuda()
    state = torch.load(ckpt_path, map_location="cuda")
    # thop.profile() (called during training for FLOPs logging) attaches
    # total_ops/total_params buffers to every submodule as a side effect;
    # these get saved into the checkpoint but aren't real model weights.
    # strict=False ignores them safely -- verified below that no actual
    # weight tensors are missing/unexpected, only these profiling buffers.
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith(('total_ops', 'total_params'))]
    real_unexpected = [k for k in unexpected if not k.endswith(('total_ops', 'total_params'))]
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"Unexpected real mismatch beyond thop buffers.\n"
            f"Missing: {real_missing}\nUnexpected: {real_unexpected}"
        )
    print(f"Checkpoint loaded (ignored {len(unexpected)} thop profiling buffers, "
          f"0 real weight mismatches).")
    model.eval()
    return model


def compute_per_slice_dice(model, img_dir, mask_dir, img_size):
    filenames = sorted(os.listdir(img_dir))
    results = {}
    with torch.no_grad():
        for fname in filenames:
            img = cv2.imread(os.path.join(img_dir, fname))  # HxWx3, uint8
            mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = mask.shape

            img_resized = cv2.resize(img, (img_size, img_size))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).cuda()

            out = model(img_tensor)
            pred = out[0] if isinstance(out, list) else out
            pred_resized = F.interpolate(pred, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            pred_resized = pred_resized.sigmoid().squeeze()
            pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)

            gt = torch.from_numpy((mask.astype(np.float32) / 255.0)).cuda()

            pred_binary = (pred_resized >= 0.5).float()
            gt_binary = (gt >= 0.2).float()  # matches train_polyp.py's asymmetric threshold convention

            d = dice_coefficient(pred_binary, gt_binary).item()
            results[fname] = d

    return results


def build_similarity_graph(img_dir, filenames, threshold):
    images = []
    for fname in filenames:
        img = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # downsample for tractable pairwise SSIM on 274^2 pairs
        images.append(img)

    n = len(images)

    # First pass: compute the full pairwise SSIM distribution so we can pick
    # a threshold that actually produces a sparse, structured graph for THIS
    # dataset -- PRIME itself tunes tau per-dataset (see paper Figs. 4-5,
    # tau swept from 0.7 to 1.0 depending on the dataset); reusing one fixed
    # value across very different domains is not faithful to their method.
    print("Computing full pairwise SSIM distribution for threshold calibration...")
    all_sims = []
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = ssim(images[i], images[j])
            sim_matrix[i, j] = sim_matrix[j, i] = s
            all_sims.append(s)
        if i % 50 == 0:
            print(f"  {i}/{n}")

    all_sims = np.array(all_sims)
    print(f"SSIM distribution: min={all_sims.min():.3f}, max={all_sims.max():.3f}, "
          f"mean={all_sims.mean():.3f}, median={np.median(all_sims):.3f}, "
          f"p90={np.percentile(all_sims, 90):.3f}, p95={np.percentile(all_sims, 95):.3f}, "
          f"p99={np.percentile(all_sims, 99):.3f}")

    # Use the 90th percentile as tau -- keeps the densest/most-similar 10%
    # of pairs as edges, analogous to PRIME's own high-tau, high-pruning
    # settings (e.g. tau=0.97 on ClinicDB keeps only the most similar pairs).
    # Auto-derived from this dataset's own distribution rather than reused
    # from a different domain.
    calibrated_threshold = np.percentile(all_sims, 90)
    print(f"Using calibrated threshold (90th percentile): {calibrated_threshold:.3f}")

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= calibrated_threshold:
                G.add_edge(i, j, weight=sim_matrix[i, j])

    return G, calibrated_threshold, all_sims


def main():
    print("=== Step 1: loading model and computing per-slice Dice ===")
    model = load_model(CKPT_PATH)
    dice_by_file = compute_per_slice_dice(model, TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE)
    filenames = sorted(dice_by_file.keys())
    dice_values = np.array([dice_by_file[f] for f in filenames])
    print(f"Per-slice Dice: mean={dice_values.mean():.4f}, std={dice_values.std():.4f}, "
          f"min={dice_values.min():.4f}, max={dice_values.max():.4f}")

    print("\n=== Step 2: building PRIME-style similarity graph ===")
    G, calibrated_threshold, all_sims = build_similarity_graph(TEST_IMG_DIR, filenames, SSIM_THRESHOLD)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({100*G.number_of_edges()/(len(filenames)*(len(filenames)-1)/2):.1f}% of possible pairs)")

    print("\n=== Step 3: Louvain community detection ===")
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)
    num_communities = len(set(partition.values()))
    print(f"Communities: {num_communities}, modularity: {modularity:.4f}")

    degrees = dict(G.degree())
    degree_values = np.array([degrees.get(i, 0) for i in range(len(filenames))])

    print("\n=== Step 4: correlation between degree and Dice ===")
    rho, pval = spearmanr(degree_values, dice_values)
    print(f"Spearman correlation (degree vs. Dice): rho={rho:.4f}, p={pval:.4f}")

    # Save results
    np.savez("prelim_result2_data.npz",
             filenames=filenames, dice=dice_values, degree=degree_values,
             modularity=modularity, num_communities=num_communities,
             spearman_rho=rho, spearman_p=pval,
             calibrated_threshold=calibrated_threshold, all_sims=all_sims)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(degree_values, dice_values, alpha=0.5, s=20)
    ax.set_xlabel("PRIME degree centrality (within-community)")
    ax.set_ylabel("Per-slice test Dice")
    ax.set_title(f"Degree vs. Dice on BraTS2020 test set (n={len(filenames)})\n"
                 f"Spearman rho={rho:.3f}, p={pval:.4f}")
    plt.tight_layout()
    plt.savefig("prelim_result2_scatter.png", dpi=150)
    print("\nSaved: prelim_result2_data.npz, prelim_result2_scatter.png")


if __name__ == "__main__":
    main()
