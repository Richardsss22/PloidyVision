#!/usr/bin/env python3
"""Cell-cycle pipeline for FBIB datasets.

This version is intentionally MATLAB-portable:
- segmentation uses the blue/DAPI channel
- phase extraction uses robust red/green scores per nucleus
- results are benchmarked against the professor ground-truth image

The key observation in these datasets is that:
- `teste*.TIF` contains R/G fluorescence + B/DAPI
- `ground-truth*.TIF` is pixel-identical to the R/G channels of `teste*.TIF`

So the most reliable route is:
1. segment nuclei from the blue channel
2. clean and quantify red/green fluorescence inside each nucleus
3. classify each nucleus from the purified R/G signal
4. compare those labels with the labels extracted from the GT file

Usage:
    python3 cell_cycle_pipeline.py \
        --data-root "/Users/ricardo/DevApps/FBIB/New Project/dados-2" \
        --output-dir results
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPS = 1e-8


@dataclass
class SegmentationConfig:
    min_area: int = 300
    max_area: int = 20000
    min_circularity: float = 0.40
    min_solidity: float = 0.75
    min_extent: float = 0.30
    min_intensity_frac: float = 0.30
    opening_size: int = 5
    closing_size: int = 5
    peak_size: int = 17
    peak_floor: float = 2.0
    border_margin: int = 5
    min_border_fraction: float = 0.40


@dataclass
class ColorConfig:
    bleed: float = 0.30
    score_percentile: float = 90.0
    gt_red_threshold: float = 0.52
    gt_green_threshold: float = 0.48
    min_total_signal: float = 0.01
    nbr_radius: int = 0


@dataclass
class DatasetPaths:
    name: str
    test_path: Path
    gt_path: Path


@dataclass
class NucleusFeatures:
    area: float
    perimeter: float
    circularity: float
    solidity: float
    extent: float
    mean_dapi: float
    int_dapi: float
    int_nobg: float
    mean_nobg: float
    std_dapi: float
    centroid_x: float
    centroid_y: float


@dataclass
class AnalysisResult:
    dataset: str
    labels: np.ndarray
    pre: Dict[str, np.ndarray]
    features: List[NucleusFeatures]
    gt_labels: np.ndarray
    baseline_labels: np.ndarray
    reference_labels: np.ndarray
    baseline_accuracy: float
    reference_fit_accuracy: float
    reference_lodo_accuracy: float
    coverage: float
    n_total: int
    n_valid: int
    n_ambiguous: int
    rg_identical: bool
    figure_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell-cycle extraction pipeline")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to dados-2")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for CSV/report/figures")
    return parser.parse_args()


def zscore(values: Sequence[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    std = float(np.std(vec))
    if std < EPS:
        return np.zeros_like(vec)
    return (vec - float(np.mean(vec))) / std


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def find_datasets(data_root: Path) -> List[DatasetPaths]:
    datasets: List[DatasetPaths] = []
    for sub in sorted(data_root.glob("sub*")):
        test_path = next(sub.glob("teste*.TIF"), None)
        gt_path = next(sub.glob("ground-truth*.TIF"), None)
        if test_path is None or gt_path is None:
            continue
        datasets.append(DatasetPaths(name=sub.name, test_path=test_path, gt_path=gt_path))
    if not datasets:
        raise FileNotFoundError(f"No datasets found under {data_root}")
    return datasets


def preprocess_dapi(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    blue = image_rgb[:, :, 2].astype(np.float32) / 255.0
    background = cv2.GaussianBlur(blue, (0, 0), 50)
    nobg = blue - background
    nobg -= float(nobg.min())
    nobg /= float(nobg.max()) + EPS
    smoothed = cv2.GaussianBlur(nobg, (0, 0), 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    return {
        "raw": blue,
        "background": background,
        "nobg": nobg,
        "smoothed": smoothed,
        "enhanced": enhanced,
    }


def threshold_and_clean(enhanced: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    enhanced_u8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(enhanced_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = enhanced_u8 > otsu_thr
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.opening_size, cfg.opening_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.closing_size, cfg.closing_size))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close) > 0
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def connected_component_props(labels: np.ndarray, intensity: np.ndarray) -> Iterable[Tuple[int, NucleusFeatures]]:
    count, label_map, stats, centroids = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8), 8)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        region = label_map == label_id
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = max(float(cv2.arcLength(contour, True)), EPS)
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), EPS)
        extent = float(area / (w * h + EPS))
        mask_values = intensity[region]
        feature = NucleusFeatures(
            area=float(area),
            perimeter=perimeter,
            circularity=circularity,
            solidity=float(area / hull_area),
            extent=extent,
            mean_dapi=float(np.mean(mask_values)),
            int_dapi=float(np.sum(mask_values)),
            int_nobg=0.0,
            mean_nobg=0.0,
            std_dapi=float(np.std(mask_values)),
            centroid_x=float(centroids[label_id, 0]),
            centroid_y=float(centroids[label_id, 1]),
        )
        yield label_id, feature


def quality_filter(mask: np.ndarray, raw_dapi: np.ndarray, nobg_dapi: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    labels = np.where(mask, 1, 0).astype(np.uint8)
    count, label_map, stats, _ = cv2.connectedComponentsWithStats(labels, 8)
    keep = np.zeros_like(label_map, dtype=bool)
    foreground = raw_dapi[mask]
    global_mean = float(np.mean(foreground)) if foreground.size else float(np.mean(raw_dapi))
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < cfg.min_area or area > cfg.max_area:
            continue
        region = label_map == label_id
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = max(float(cv2.arcLength(contour, True)), EPS)
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), EPS)
        extent = float(area / (w * h + EPS))
        solidity = float(area / hull_area)
        mean_intensity = float(np.mean(raw_dapi[region]))
        if circularity < cfg.min_circularity:
            continue
        if solidity < cfg.min_solidity:
            continue
        if extent < cfg.min_extent:
            continue
        if mean_intensity < global_mean * cfg.min_intensity_frac:
            continue
        keep |= region
    return keep


def watershed_split(mask: np.ndarray, enhanced: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    distance = ndi.distance_transform_edt(mask)
    distance_sm = ndi.gaussian_filter(distance, sigma=2.0)
    positive = distance_sm[distance_sm > 0]
    if positive.size == 0:
        return np.zeros_like(mask, dtype=np.int32)
    adaptive_floor = max(cfg.peak_floor, float(np.percentile(positive, 65)))
    local_max = (distance_sm == maximum_filter(distance_sm, size=cfg.peak_size)) & (distance_sm >= adaptive_floor) & mask
    markers, _ = ndi.label(local_max)
    markers = markers.astype(np.int32) + 1
    markers[~mask] = 0
    color = np.dstack([np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)] * 3)
    labels = cv2.watershed(color, markers)
    labels[(labels <= 1) | (~mask)] = 0
    _, labels = cv2.connectedComponents((labels > 0).astype(np.uint8))
    return labels.astype(np.int32)


def soft_border_filter(labels: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label_id in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        touches = y1 == 0 or x1 == 0 or y2 == h or x2 == w
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        dmin = min(cy, cx, h - cy, w - cx)
        bbox_area = float((y2 - y1) * (x2 - x1))
        inside_fraction = float((min(x2, w) - max(x1, 0)) * (min(y2, h) - max(y1, 0)) / (bbox_area + EPS))
        if (not touches) or dmin >= cfg.border_margin or inside_fraction >= cfg.min_border_fraction:
            out[labels == label_id] = next_id
            next_id += 1
    return out


def segment_nuclei(image_rgb: np.ndarray, cfg: SegmentationConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    pre = preprocess_dapi(image_rgb)
    coarse = threshold_and_clean(pre["enhanced"], cfg)
    filtered = quality_filter(coarse, pre["raw"], pre["nobg"], cfg)
    labels = watershed_split(filtered, pre["enhanced"], cfg)
    labeled_pixels = int((labels > 0).sum())
    if labeled_pixels < 0.5 * int(filtered.sum()):
        _, labels = cv2.connectedComponents(filtered.astype(np.uint8))
    if int(labels.max()) == 0:
        _, labels = cv2.connectedComponents(filtered.astype(np.uint8))
    labels = soft_border_filter(labels, cfg)
    return labels, pre


def extract_features(labels: np.ndarray, pre: Dict[str, np.ndarray]) -> List[NucleusFeatures]:
    features: List[NucleusFeatures] = []
    raw = pre["raw"]
    nobg = pre["nobg"]
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not np.any(region):
            continue
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(np.sum(region))
        perimeter = max(float(cv2.arcLength(contour, True)), EPS)
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), EPS)
        features.append(
            NucleusFeatures(
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                solidity=float(area / hull_area),
                extent=float(area / (w * h + EPS)),
                mean_dapi=float(np.mean(raw[region])),
                int_dapi=float(np.sum(raw[region])),
                int_nobg=float(np.sum(nobg[region])),
                mean_nobg=float(np.mean(nobg[region])),
                std_dapi=float(np.std(raw[region])),
                centroid_x=float(np.mean(np.where(region)[1])),
                centroid_y=float(np.mean(np.where(region)[0])),
            )
        )
    return features


def purify_rg(image_rgb: np.ndarray, mask: np.ndarray, cfg: ColorConfig) -> Tuple[np.ndarray, np.ndarray]:
    red = image_rgb[:, :, 0].astype(np.float32) / 255.0
    green = image_rgb[:, :, 1].astype(np.float32) / 255.0
    outside = ~mask
    if np.any(outside):
        red = np.clip(red - float(np.median(red[outside])), 0.0, None)
        green = np.clip(green - float(np.median(green[outside])), 0.0, None)
    red_pure = np.clip(red - cfg.bleed * green, 0.0, None)
    green_pure = np.clip(green - cfg.bleed * red, 0.0, None)
    if np.any(mask):
        red_scale = max(float(np.percentile(red_pure[mask], 99)), EPS)
        green_scale = max(float(np.percentile(green_pure[mask], 99)), EPS)
        red_pure = np.clip(red_pure / red_scale, 0.0, 1.0)
        green_pure = np.clip(green_pure / green_scale, 0.0, 1.0)
    return red_pure, green_pure


def nucleus_mask(labels: np.ndarray, feature: NucleusFeatures, label_id: int, radius: int) -> np.ndarray:
    region = labels == label_id
    if radius <= 0:
        return region
    yy, xx = np.ogrid[: labels.shape[0], : labels.shape[1]]
    disk_mask = ((xx - feature.centroid_x) ** 2 + (yy - feature.centroid_y) ** 2) <= radius * radius
    return region & disk_mask


def compute_rg_labels_direct(labels: np.ndarray, features: List[NucleusFeatures], image_rgb: np.ndarray, cfg: ColorConfig) -> Tuple[np.ndarray, np.ndarray]:
    red_pure, green_pure = purify_rg(image_rgb, labels > 0, cfg)
    label_values = []
    ratio_values = []
    for idx, feature in enumerate(features, start=1):
        region = nucleus_mask(labels, feature, idx, cfg.nbr_radius)
        if np.sum(region) < 5:
            label_values.append(np.nan)
            ratio_values.append(np.nan)
            continue
        red_score = float(np.percentile(red_pure[region], cfg.score_percentile))
        green_score = float(np.percentile(green_pure[region], cfg.score_percentile))
        total = red_score + green_score
        if total < cfg.min_total_signal:
            label_values.append(np.nan)
            ratio_values.append(np.nan)
            continue
        ratio = red_score / (total + EPS)
        ratio_values.append(ratio)
        if ratio >= cfg.gt_red_threshold:
            label_values.append(1.0)
        elif ratio <= cfg.gt_green_threshold:
            label_values.append(2.0)
        else:
            label_values.append(np.nan)
    return np.asarray(label_values, dtype=np.float64), np.asarray(ratio_values, dtype=np.float64)


def classify_from_dapi(features: List[NucleusFeatures]) -> np.ndarray:
    if not features:
        return np.zeros(0, dtype=np.int32)
    X = np.column_stack(
        [
            zscore([math.log(max(f.int_nobg, EPS)) for f in features]),
            zscore([f.area for f in features]),
            zscore([f.mean_nobg for f in features]),
        ]
    )
    model = KMeans(n_clusters=2, random_state=42, n_init=20, max_iter=500)
    raw_labels = model.fit_predict(X)
    dna = np.array([f.int_dapi for f in features], dtype=np.float64)
    ordered = np.zeros_like(raw_labels, dtype=np.int32)
    means = []
    for cluster_id in (0, 1):
        vals = dna[raw_labels == cluster_id]
        means.append(float(np.mean(vals)) if vals.size else -1.0)
    rank = np.argsort(means)
    for new_label, cluster_id in enumerate(rank, start=1):
        ordered[raw_labels == cluster_id] = new_label
    return ordered


def feature_vector_area_dapi(feature: NucleusFeatures) -> np.ndarray:
    log_int = math.log(max(feature.int_nobg, EPS))
    log_area = math.log(max(feature.area, EPS))
    return np.array([log_int, log_area], dtype=np.float64)


def fit_reference_knn(X: np.ndarray, y: np.ndarray, k: int = 7) -> Dict[str, np.ndarray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < EPS] = 1.0
    model = {
        "mean": mean,
        "std": std,
        "X": (X - mean) / std,
        "y": y.astype(np.int32),
        "k": np.array([min(k, len(X))], dtype=np.int32),
    }
    return model


def predict_reference_knn(model: Dict[str, np.ndarray], X_query: np.ndarray) -> np.ndarray:
    Xs = model["X"]
    ys = model["y"]
    mean = model["mean"]
    std = model["std"]
    k = int(model["k"][0])
    q = (X_query - mean) / std
    out = np.zeros(len(q), dtype=np.int32)
    for i, row in enumerate(q):
        dists = np.sqrt(np.sum((Xs - row) ** 2, axis=1))
        idx = np.argsort(dists)[:k]
        votes = {1: 0.0, 2: 0.0}
        for j in idx:
            weight = 1.0 / (float(dists[j]) + 1e-6)
            votes[int(ys[j])] += weight
        out[i] = 1 if votes[1] >= votes[2] else 2
    return out


def accuracy_against_gt(pred: np.ndarray, gt_labels: np.ndarray) -> Tuple[float, int, int, int]:
    valid = np.isfinite(gt_labels)
    n_valid = int(np.sum(valid))
    n_ambiguous = int(np.sum(~valid))
    if n_valid == 0:
        return 0.0, 0, 0, n_ambiguous
    correct = int(np.sum(pred[valid] == gt_labels[valid]))
    return correct / n_valid, correct, n_valid, n_ambiguous


def phase_colors() -> Dict[int, Tuple[float, float, float]]:
    return {
        1: (0.92, 0.22, 0.22),
        2: (0.12, 0.72, 0.30),
    }


def draw_overlay(ax: plt.Axes, base_gray: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray, title: str) -> None:
    ax.imshow(base_gray, cmap="gray")
    colors = phase_colors()
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not np.any(region):
            continue
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = contours[0][:, 0, :]
        label_value = pred_labels[label_id - 1]
        if not np.isfinite(label_value):
            color = (1.0, 1.0, 1.0)
        else:
            color = colors.get(int(label_value), (1.0, 1.0, 1.0))
        ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_axis_off()


def save_figure(
    dataset: str,
    pre: Dict[str, np.ndarray],
    labels: np.ndarray,
    gt_labels: np.ndarray,
    baseline_labels: np.ndarray,
    reference_labels: np.ndarray,
    gt_img: np.ndarray,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    mask = labels > 0
    red_pure, green_pure = purify_rg(gt_img, mask, ColorConfig())
    clean_gt = np.zeros_like(gt_img, dtype=np.float32)
    clean_gt[:, :, 0] = red_pure
    clean_gt[:, :, 1] = green_pure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(pre["raw"], cmap="gray")
    axes[0, 0].set_title(f"{dataset} - Raw DAPI")
    axes[0, 0].set_axis_off()
    axes[0, 1].imshow(pre["enhanced"], cmap="gray")
    axes[0, 1].set_title("Enhanced DAPI")
    axes[0, 1].set_axis_off()
    axes[0, 2].imshow(labels > 0, cmap="gray")
    axes[0, 2].set_title(f"Binary mask ({int(labels.max())} nuclei)")
    axes[0, 2].set_axis_off()
    draw_overlay(axes[1, 0], pre["raw"], labels, gt_labels, "GT labels per nucleus")
    draw_overlay(axes[1, 1], pre["raw"], labels, baseline_labels, "Baseline: DAPI k-means")
    draw_overlay(axes[1, 2], pre["raw"], labels, reference_labels, "Reference: DAPI+Area kNN")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_global_scatter(results: List[AnalysisResult], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = phase_colors()
    for result in results:
        pts = []
        lbls = []
        for feature, gt_label in zip(result.features, result.gt_labels):
            if not np.isfinite(gt_label):
                continue
            pts.append(feature_vector_area_dapi(feature))
            lbls.append(int(gt_label))
        if not pts:
            continue
        X = np.asarray(pts)
        y = np.asarray(lbls)
        for cls in (1, 2):
            mask = y == cls
            if np.any(mask):
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=28,
                    alpha=0.65,
                    color=colors[cls],
                    label=f"{result.dataset} - {'G1' if cls == 1 else 'G2/M'}",
                )
    ax.set_xlabel("log(IntNoBg)")
    ax.set_ylabel("log(Area)")
    ax.set_title("DAPI vs Area Across All Labeled Nuclei")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyse_dataset(dataset: DatasetPaths, output_dir: Path) -> AnalysisResult:
    seg_cfg = SegmentationConfig()
    color_cfg = ColorConfig()
    test_img = load_rgb(dataset.test_path)
    gt_img = load_rgb(dataset.gt_path)
    labels, pre = segment_nuclei(test_img, seg_cfg)
    features = extract_features(labels, pre)

    def local_rg_labels(image_rgb: np.ndarray) -> np.ndarray:
        red_pure, green_pure = purify_rg(image_rgb, labels > 0, color_cfg)
        out = []
        for idx, feature in enumerate(features, start=1):
            region = nucleus_mask(labels, feature, idx, color_cfg.nbr_radius)
            if np.sum(region) < 5:
                out.append(np.nan)
                continue
            red_score = float(np.percentile(red_pure[region], color_cfg.score_percentile))
            green_score = float(np.percentile(green_pure[region], color_cfg.score_percentile))
            total = red_score + green_score
            if total < color_cfg.min_total_signal:
                out.append(np.nan)
                continue
            ratio = red_score / (total + EPS)
            if ratio >= color_cfg.gt_red_threshold:
                out.append(1.0)
            elif ratio <= color_cfg.gt_green_threshold:
                out.append(2.0)
            else:
                out.append(np.nan)
        return np.asarray(out, dtype=np.float64)

    gt_labels = local_rg_labels(gt_img)
    baseline_labels = classify_from_dapi(features)
    _, _, n_valid, n_ambiguous = accuracy_against_gt(baseline_labels, gt_labels)
    baseline_accuracy, _, _, _ = accuracy_against_gt(baseline_labels, gt_labels)
    figure_path = output_dir / "figures" / f"{dataset.name}_summary.png"
    return AnalysisResult(
        dataset=dataset.name,
        labels=labels,
        pre=pre,
        features=features,
        gt_labels=gt_labels,
        baseline_labels=baseline_labels.astype(np.float64),
        reference_labels=np.full(len(features), np.nan, dtype=np.float64),
        baseline_accuracy=baseline_accuracy,
        reference_fit_accuracy=0.0,
        reference_lodo_accuracy=0.0,
        coverage=n_valid / max(len(features), 1),
        n_total=len(features),
        n_valid=n_valid,
        n_ambiguous=n_ambiguous,
        rg_identical=bool(np.array_equal(test_img[:, :, :2], gt_img[:, :, :2])),
        figure_path=figure_path,
    )


def benchmark_reference_model(results: List[AnalysisResult]) -> Dict[str, np.ndarray]:
    rows = []
    groups = []
    for result in results:
        for feature, gt_label in zip(result.features, result.gt_labels):
            if not np.isfinite(gt_label):
                continue
            rows.append(feature_vector_area_dapi(feature))
            groups.append(result.dataset)
    X_all = np.asarray(rows, dtype=np.float64)
    y_all = np.asarray([int(gt) for result in results for gt in result.gt_labels if np.isfinite(gt)], dtype=np.int32)
    group_all = np.asarray(groups)

    fit_model = fit_reference_knn(X_all, y_all, k=7)
    fit_pred = predict_reference_knn(fit_model, X_all)

    offset = 0
    for result in results:
        valid_mask = np.isfinite(result.gt_labels)
        n_valid = int(np.sum(valid_mask))
        pred_slice = fit_pred[offset : offset + n_valid]
        offset += n_valid
        result.reference_labels = np.full(len(result.features), np.nan, dtype=np.float64)
        result.reference_labels[valid_mask] = pred_slice.astype(np.float64)
        result.reference_fit_accuracy, _, _, _ = accuracy_against_gt(result.reference_labels, result.gt_labels)

    for holdout in sorted(set(group_all.tolist())):
        train_mask = group_all != holdout
        test_mask = group_all == holdout
        model = fit_reference_knn(X_all[train_mask], y_all[train_mask], k=7)
        holdout_pred = predict_reference_knn(model, X_all[test_mask])
        cursor = 0
        for result in results:
            if result.dataset != holdout:
                continue
            valid_mask = np.isfinite(result.gt_labels)
            n_valid = int(np.sum(valid_mask))
            pred_slice = holdout_pred[cursor : cursor + n_valid]
            cursor += n_valid
            lodo_labels = np.full(len(result.features), np.nan, dtype=np.float64)
            lodo_labels[valid_mask] = pred_slice.astype(np.float64)
            result.reference_lodo_accuracy, _, _, _ = accuracy_against_gt(lodo_labels, result.gt_labels)

    return fit_model


def finalize_figures(results: List[AnalysisResult], data_root: Path, output_dir: Path) -> None:
    dataset_map = {d.name: d for d in find_datasets(data_root)}
    for result in results:
        gt_img = load_rgb(dataset_map[result.dataset].gt_path)
        save_figure(
            result.dataset,
            result.pre,
            result.labels,
            result.gt_labels,
            result.baseline_labels,
            result.reference_labels,
            gt_img,
            result.figure_path,
        )
    save_global_scatter(results, output_dir / "figures" / "global_dapi_area_scatter.png")


def save_reference_artifacts(results: List[AnalysisResult], model: Dict[str, np.ndarray], output_dir: Path) -> None:
    np.savez(
        output_dir / "reference_knn_model.npz",
        mean=model["mean"],
        std=model["std"],
        X=model["X"],
        y=model["y"],
        k=model["k"],
    )
    with (output_dir / "reference_knn_training_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "log_int_nobg", "log_area", "gt_label"])
        writer.writeheader()
        for result in results:
            for feature, gt_label in zip(result.features, result.gt_labels):
                if not np.isfinite(gt_label):
                    continue
                vec = feature_vector_area_dapi(feature)
                writer.writerow(
                    {
                        "dataset": result.dataset,
                        "log_int_nobg": f"{vec[0]:.6f}",
                        "log_area": f"{vec[1]:.6f}",
                        "gt_label": int(gt_label),
                    }
                )


def write_csv(results: List[AnalysisResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "n_total",
                "n_valid",
                "n_ambiguous",
                "coverage",
                "baseline_accuracy",
                "reference_fit_accuracy",
                "reference_lodo_accuracy",
                "rg_identical_to_gt",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "dataset": result.dataset,
                    "n_total": result.n_total,
                    "n_valid": result.n_valid,
                    "n_ambiguous": result.n_ambiguous,
                    "coverage": f"{result.coverage:.4f}",
                    "baseline_accuracy": f"{result.baseline_accuracy:.4f}",
                    "reference_fit_accuracy": f"{result.reference_fit_accuracy:.4f}",
                    "reference_lodo_accuracy": f"{result.reference_lodo_accuracy:.4f}",
                    "rg_identical_to_gt": str(result.rg_identical),
                }
            )


def write_report(results: List[AnalysisResult], report_path: Path, data_root: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    avg_baseline = float(np.mean([r.baseline_accuracy for r in results]))
    avg_fit = float(np.mean([r.reference_fit_accuracy for r in results]))
    avg_lodo = float(np.mean([r.reference_lodo_accuracy for r in results]))
    avg_cov = float(np.mean([r.coverage for r in results]))
    lines: List[str] = []
    lines.append("# Cell-Cycle Results")
    lines.append("")
    lines.append(f"Data root: `{data_root}`")
    lines.append("")
    lines.append("## Correct Constraint")
    lines.append("")
    lines.append("The final prediction must be produced from `teste*.TIF` without reading the ground truth.")
    lines.append("So the valid pipeline is DAPI-only prediction plus GT-only evaluation.")
    lines.append("")
    lines.append("The fact that `teste` and `ground-truth` share the same red/green signal is useful only to extract evaluation labels per nucleus.")
    lines.append("")
    lines.append("## DAPI vs Area Strategy")
    lines.append("")
    lines.append("The best practical predictor in this workspace is a weighted kNN atlas in the 2D feature space:")
    lines.append("- `x = log(IntNoBg)`")
    lines.append("- `y = log(Area)`")
    lines.append("")
    lines.append("This is deliberately simple and MATLAB-portable.")
    lines.append("")
    lines.append("## Benchmark")
    lines.append("")
    lines.append("| Dataset | Nuclei | Valid GT | Coverage | Baseline k-means | Reference fit | Leave-one-dataset-out | Figure |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        lines.append(
            f"| {result.dataset} | {result.n_total} | {result.n_valid} | {100*result.coverage:.1f}% | "
            f"{100*result.baseline_accuracy:.1f}% | {100*result.reference_fit_accuracy:.1f}% | "
            f"{100*result.reference_lodo_accuracy:.1f}% | [figure]({result.figure_path.as_posix()}) |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Mean baseline k-means accuracy: **{100*avg_baseline:.1f}%**")
    lines.append(f"- Mean reference-fit accuracy: **{100*avg_fit:.1f}%**")
    lines.append(f"- Mean leave-one-dataset-out accuracy: **{100*avg_lodo:.1f}%**")
    lines.append(f"- Mean GT coverage: **{100*avg_cov:.1f}%**")
    lines.append("")
    lines.append("## Honest Conclusion")
    lines.append("")
    lines.append("- If the reference atlas is trained and evaluated on the same annotated nuclei, the DAPI+Area model reaches very high performance.")
    lines.append("- If one whole dataset is held out, performance drops a lot, which means generalization is still the real bottleneck.")
    lines.append("- `sub4` remains the hardest dataset because DAPI and area overlap much more between classes.")
    lines.append("")
    lines.append("## MATLAB Port Notes")
    lines.append("")
    lines.append("- Keep the DAPI segmentation pipeline in the blue channel.")
    lines.append("- Use GT only inside the evaluation function.")
    lines.append("- Do not use the red/green channels from `teste*.TIF` in the final predictor.")
    lines.append("- Reference predictor to port first: weighted kNN on `[log(int_nobg), log(area)]`.")
    lines.append("- In MATLAB, standardize the two features with training mean/std, compute Euclidean distance to all training nuclei, keep the 7 nearest, and vote with weight `1/(dist+1e-6)`.")
    lines.append("- Use the baseline k-means only as fallback or comparison.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = find_datasets(args.data_root)
    results = [analyse_dataset(dataset, output_dir) for dataset in datasets]
    reference_model = benchmark_reference_model(results)
    finalize_figures(results, args.data_root.resolve(), output_dir)
    save_reference_artifacts(results, reference_model, output_dir)
    write_csv(results, output_dir / "cell_cycle_grid_results.csv")
    write_report(results, output_dir / "cell_cycle_report.md", args.data_root.resolve())
    print("Saved:")
    print(output_dir / "cell_cycle_grid_results.csv")
    print(output_dir / "cell_cycle_report.md")
    print(output_dir / "reference_knn_model.npz")
    for result in results:
        print(
            f"{result.dataset}: baseline={100*result.baseline_accuracy:.1f}% "
            f"fit={100*result.reference_fit_accuracy:.1f}% "
            f"lodo={100*result.reference_lodo_accuracy:.1f}% "
            f"coverage={100*result.coverage:.1f}%"
        )


if __name__ == "__main__":
    main()
