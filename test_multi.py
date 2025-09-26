# from utils.attention_sdxl_pipeline import AttentionStableDiffusionXLPipeline
from utils.counting_attention_guidance_sdxl_pipeline import AttentionGuidanceStableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore, LossGuideAttentionStore
import os
import torch
import random
import numpy as np
from typing import List, Union
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from skimage.draw import disk
from scipy.spatial.distance import cdist
import diffusers
import inflect
import json

p = inflect.engine()

PIPELINE_PATH = "/data/wsq_data/stable-diffusion-xl-base-1/"
DEVICE = "cuda:0"

NUMERAL_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


def set_seed(seed: int) -> torch.Generator:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return gen


def save_latents_as_image(latents, save_path):
    """
    latents: Tensor or np.array of shape (C, H, W) or (1, C, H, W)
    save_path: path to save the output image (e.g., PNG)
    """
    if hasattr(latents, "detach"):  # PyTorch Tensor
        latents = latents.detach().cpu().numpy()

    if latents.ndim == 4:
        latents = latents[0]  # remove batch dim

    # Reduce C-dim: mean over channels (you can change to latents[0] for a single channel)
    latents_vis = latents.mean(axis=0)  # (H, W)

    # Normalize to [0,1]
    latents_vis -= latents_vis.min()
    latents_vis /= (latents_vis.max() + 1e-8)

    plt.imshow(latents_vis, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_attention_map(att_map, steps=1):
    """
    norm and square，最大最小值归一化
    """
    for _ in range(steps):
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        att_map = att_map ** 2
    # 最后再归一化一次
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map


def noise_recreate(
        latents,
        attention_map,
        numeral,
        alpha_background=0.98,
        alpha_content=1.05,
        save_dir="./vis_output",
        prefix="sample",
        use_patch_mode=True,  # 新增参数：是否使用patch模式
):
    """ """
    os.makedirs(save_dir, exist_ok=True)
    new_latents = latents.clone().detach()

    def cluster_high_attention_regions(att_map, numeral=5, percentile=70, threshold=0.6):
        """
        设置threshold先做一遍过滤，然后使用 KMeans 聚类找到最显著的 numeral 个区域（cluster_masks） 和centers

        """
        h, w = att_map.shape

        # 1. 扁平化和计算阈值（只保留高 attention 区域）
        values = att_map.flatten()
        # threshold = 0.5  # 可以改为 np.percentile(values, percentile) 进行自适应阈值，这里需要实验
        high_att_mask = values >= threshold
        high_att_mask_2d = high_att_mask.reshape(att_map.shape)

        # 2. 只保留高 attention 区域的坐标和值
        coords = np.array([(i, j) for i in range(h) for j in range(w)])
        coords = coords[high_att_mask]
        weights = values[high_att_mask]

        # 3. 归一化权重
        weights = weights / (weights.sum() + 1e-8)

        # 4. 聚类（KMeans）
        if len(coords) < numeral:
            # 少于 numeral 个点，就直接返回这些点作为中心
            centers = coords
            labels = np.arange(len(coords))  # 每个点一个 cluster
        else:
            kmeans = KMeans(n_clusters=numeral, random_state=42)
            kmeans.fit(coords, sample_weight=weights)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

        # 5. 返回整型像素坐标
        centers = np.round(centers).astype(int)

        # 6. 为每个 center 构造 mask
        cluster_masks = []
        for i in range(numeral):
            mask = np.zeros((h, w), dtype=bool)
            cluster_coords = coords[labels == i]
            for y, x in cluster_coords:
                mask[y, x] = True
            cluster_masks.append(mask)

        whole_cluster_masks = []

        # 7. 为每个center构造whole cluster mask
        # 计算所有像素到所有centers的距离
        all_coords = np.array([(i, j) for i in range(h) for j in range(w)])
        distances = cdist(all_coords, centers, metric='euclidean')  # shape: (h*w, numeral)

        # 每个像素分配给最近的center
        closest_center_indices = np.argmin(distances, axis=1)

        # 为每个center创建mask
        for i in range(len(centers)):
            mask = (closest_center_indices == i).reshape((h, w))
            whole_cluster_masks.append(mask)
        # ------------------------- vis start -------------------------

        os.makedirs(save_dir, exist_ok=True)

        # 1. Normalized Attention Map
        # fig, ax = plt.subplots()
        # ax.imshow(att_map, cmap="viridis")
        # ax.axis("off")
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f"{prefix}_att_map.png"), dpi=300, bbox_inches='tight')
        # plt.close()
        fig, ax = plt.subplots()
        ax.imshow(att_map, cmap="viridis")
        ax.axis("off")
        plt.savefig(os.path.join(save_dir, f"{prefix}_att_map.png"),
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        # 2. Thresholded Mask
        fig, ax = plt.subplots()
        ax.imshow(high_att_mask_2d, cmap="gray")
        ax.axis("off")
        plt.savefig(os.path.join(save_dir, f"{prefix}_threshold_mask.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 3. Cluster Masks (color-coded)
        h, w = att_map.shape
        color_mask = np.zeros((h, w, 3), dtype=np.float32)
        cmap_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

        for idx, mask in enumerate(cluster_masks):
            color = mcolors.to_rgb(cmap_list[idx % len(cmap_list)])
            for c in range(3):
                color_mask[..., c] += mask * color[c]

        color_mask = np.clip(color_mask, 0, 1)

        fig, ax = plt.subplots()
        ax.imshow(color_mask)
        ax.scatter(centers[:, 1], centers[:, 0], c="white", marker="x", s=50, label="Centers")
        ax.legend()
        ax.axis("off")
        # plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_cluster_masks.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 4. Whole Cluster Masks (color-coded)
        color_mask = np.zeros((h, w, 3), dtype=np.float32)

        for idx, mask in enumerate(whole_cluster_masks):
            color = mcolors.to_rgb(cmap_list[idx % len(cmap_list)])
            for c in range(3):
                color_mask[..., c] += mask * color[c]

        color_mask = np.clip(color_mask, 0, 1)

        fig, ax = plt.subplots()
        ax.imshow(color_mask)
        # ax.scatter(centers[:, 1], centers[:, 0], c="white", marker="x", s=50, label="Centers")
        # 在每个中心位置标上对应的 idx
        for idx, (y, x) in enumerate(centers):
            ax.text(x, y, str(idx), color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        # ax.legend()
        ax.axis("off")
        # plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_whole_cluster_masks.png"), dpi=300, bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        # ------------------------- vis end -------------------------

        return centers, high_att_mask_2d, cluster_masks, whole_cluster_masks

    def generate_compact_mask(center, cluster_mask, min_area=None, area_ratio=4):
        """
        根据cluster_mask面积，对其进行压缩处理，产生一个更紧凑的 mask
        1. 计算cluster_mask的有效面积
        2. 围绕center构建一个圆形区域，半径根据面积计算
        Args:
            center: tuple (cy, cx)，压缩中心位置
            cluster_mask: np.ndarray，原始 bool 或二值 mask
        Returns:
            compact_mask: np.ndarray，同尺寸布尔类型的紧凑 mask
        """
        # 获取原始mask的尺寸
        h, w = cluster_mask.shape

        # 计算原始mask的有效面积(即前景像素数量)
        area = np.sum(cluster_mask)
        area = area / area_ratio  # 将面积缩小到1/？，避免过大圆形遮挡过多区域
        # 如果面积为0，返回全零mask
        if area == 0:
            return np.zeros_like(cluster_mask, dtype=bool)

        if min_area is not None:
            # 如果提供了最小面积，则使用最小面积
            area = min(area, min_area)

        # 根据面积计算圆形区域的半径
        # 面积公式: area = π * r^2，求解半径r
        radius = int(np.sqrt(area / np.pi))

        # 确保半径至少为1
        radius = max(1, radius)
        radius = min(radius, min(h, w) // 6)  # 确保半径不超过图像尺寸的1/6(即直径不超过1/3)

        # 创建一个与原始mask相同尺寸的全零数组
        compact_mask = np.zeros_like(cluster_mask, dtype=bool)

        # 使用skimage.draw.disk函数在center位置绘制圆形区域
        cy, cx = center
        rr, cc = disk((cy, cx), radius, shape=cluster_mask.shape)
        compact_mask[rr, cc] = True

        return compact_mask

    def generate_compact_mask_and_move(center, cluster_mask, new_latents, new_processed_map, use_patch_mode=False):
        """
        根据cluster_mask面积，对其进行压缩处理，产生一个更紧凑的 mask
        新增：支持patch模式，将32x32的操作映射到128x128的4x4 patch
        """
        # 获取原始mask的尺寸
        h, w = cluster_mask.shape

        # 计算原始mask的有效面积(即前景像素数量)
        area = np.sum(cluster_mask)

        # 如果面积为0，返回全零mask
        if area == 0:
            return np.zeros_like(cluster_mask, dtype=bool)

        # 根据面积计算圆形区域的半径
        radius = int(np.sqrt(area / np.pi))

        # 创建一个与原始mask相同尺寸的全零数组
        compact_mask = np.zeros_like(cluster_mask, dtype=bool)

        # 使用skimage.draw.disk函数在center位置绘制圆形区域
        cy, cx = center
        rr, cc = disk((cy, cx), radius, shape=cluster_mask.shape)
        compact_mask[rr, cc] = True

        # 将cluster mask中的像素点压缩到这个圆形compact mask区域内（不相交的位置互换）
        # 创建排除交集的 mask
        overlap_mask = cluster_mask & compact_mask
        cluster_only = cluster_mask & (~overlap_mask)
        compact_only = compact_mask & (~overlap_mask)

        # 找出位置索引
        cluster_indices = np.argwhere(cluster_only)
        compact_indices = np.argwhere(compact_only)

        n = min(len(cluster_indices), len(compact_indices))

        # 对cluster indices做排序，优先选择对应ca值大的index（理论上cluster mask和圆compact mask面积相等，但实际上圆面积往往更小，所以要先把大ca值的latent换进来）
        cluster_values = [new_processed_map[y, x] for y, x in cluster_indices]
        # 创建索引-值对，并按值从大到小排序
        sorted_cluster_pairs = sorted(zip(cluster_indices, cluster_values), key=lambda x: -x[1])
        # 提取排序后的索引
        cluster_indices = np.array([pair[0] for pair in sorted_cluster_pairs])

        cluster_indices = cluster_indices[:n]  # 排序后的前n个索引
        compact_indices = compact_indices[:n]

        # 新增：patch模式下的latents操作
        if use_patch_mode:
            # 将32x32坐标映射到128x128的4x4 patch
            for i in range(n):
                y1, x1 = cluster_indices[i]
                y2, x2 = compact_indices[i]

                # 计算对应的4x4 patch范围
                patch_y1_start, patch_y1_end = y1 * 4, (y1 + 1) * 4
                patch_x1_start, patch_x1_end = x1 * 4, (x1 + 1) * 4
                patch_y2_start, patch_y2_end = y2 * 4, (y2 + 1) * 4
                patch_x2_start, patch_x2_end = x2 * 4, (x2 + 1) * 4

                # 交换整个4x4 patch
                tmp = new_latents[:, :, patch_y1_start:patch_y1_end, patch_x1_start:patch_x1_end].clone()
                new_latents[:, :, patch_y1_start:patch_y1_end, patch_x1_start:patch_x1_end] = \
                    new_latents[:, :, patch_y2_start:patch_y2_end, patch_x2_start:patch_x2_end]
                new_latents[:, :, patch_y2_start:patch_y2_end, patch_x2_start:patch_x2_end] = tmp

                # 为了可视化下对应操作在cross attention中是怎么样的
                tmp = new_processed_map[y1, x1].copy()
                new_processed_map[y1, x1] = new_processed_map[y2, x2]
                new_processed_map[y2, x2] = tmp
        else:
            # 原有的像素级操作
            for i in range(n):
                y1, x1 = cluster_indices[i]
                y2, x2 = compact_indices[i]

                tmp = new_latents[:, :, y1, x1].clone()
                new_latents[:, :, y1, x1] = new_latents[:, :, y2, x2]
                new_latents[:, :, y2, x2] = tmp

                # 为了可视化下对应操作在cross attention中是怎么样的
                tmp = new_processed_map[y1, x1].copy()
                new_processed_map[y1, x1] = new_processed_map[y2, x2]
                new_processed_map[y2, x2] = tmp

        # 找出 compact_mask 内所有的位置索引
        compact_ys, compact_xs = np.where(compact_mask)
        compact_positions = list(zip(compact_ys, compact_xs))
        # 计算这些位置到圆心的距离，并排序（距离越小 → 越靠近圆心）
        distances = [np.sqrt((y - cy) ** 2 + (x - cx) ** 2) for y, x in compact_positions]
        sorted_by_distance = sorted(zip(compact_positions, distances), key=lambda x: x[1])
        compact_positions_sorted = [pos for pos, _ in sorted_by_distance]
        # 对 compact 区域内的位置按 new_processed_map 值从大到小排序：
        processed_values = [new_processed_map[y, x] for y, x in compact_positions]
        sorted_by_value = sorted(zip(compact_positions, processed_values), key=lambda x: -x[1])
        value_sorted_positions = [pos for pos, _ in sorted_by_value]

        # 将 latent 值从 value 排序对应位置 赋值到 距离排序对应位置
        # 备份一份 latent，防止覆盖
        latent_backup = new_latents.clone()
        map_backup = new_processed_map.copy()

        if use_patch_mode:
            # patch模式下的重排
            for src_pos, dst_pos in zip(value_sorted_positions, compact_positions_sorted):
                y_src, x_src = src_pos
                y_dst, x_dst = dst_pos

                # 计算对应的4x4 patch范围
                patch_src_y_start, patch_src_y_end = y_src * 4, (y_src + 1) * 4
                patch_src_x_start, patch_src_x_end = x_src * 4, (x_src + 1) * 4
                patch_dst_y_start, patch_dst_y_end = y_dst * 4, (y_dst + 1) * 4
                patch_dst_x_start, patch_dst_x_end = x_dst * 4, (x_dst + 1) * 4

                new_latents[:, :, patch_dst_y_start:patch_dst_y_end, patch_dst_x_start:patch_dst_x_end] = \
                    latent_backup[:, :, patch_src_y_start:patch_src_y_end, patch_src_x_start:patch_src_x_end]
                new_processed_map[y_dst, x_dst] = map_backup[y_src, x_src]
        else:
            # 原有的像素级重排
            for src_pos, dst_pos in zip(value_sorted_positions, compact_positions_sorted):
                y_src, x_src = src_pos
                y_dst, x_dst = dst_pos
                new_latents[:, :, y_dst, x_dst] = latent_backup[:, :, y_src, x_src]
                new_processed_map[y_dst, x_dst] = map_backup[y_src, x_src]

        return compact_mask, new_latents, new_processed_map

    processed_map = attention_map  # 归一化
    new_processed_map = processed_map.clone().detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    # 聚类，并生成每个cluster的mask，以及Voronoi区域
    centers, high_att_mask_2d, cluster_masks, whole_cluster_masks = cluster_high_attention_regions(
        processed_map, numeral=numeral, threshold=0.6
    )

    # TODO: use it
    # 进一步处理每个cluster的mask，令对应latents向center塌缩聚集
    for idx, mask in enumerate(cluster_masks):
        center = tuple(map(int, centers[idx]))  # 注意 center 是 (y, x)
        compact, new_latents, new_processed_map = generate_compact_mask_and_move(
            center, mask, new_latents=new_latents, new_processed_map=new_processed_map,
            use_patch_mode=use_patch_mode
        )

    # 用于std mask
    compact_masks = []

    # # 用高亮ca对应的cluster-masks生成compact-mask，这个时候直接令compact-mask的面积等于原面积
    # for idx, mask in enumerate(cluster_masks):
    #     center = tuple(map(int, centers[idx]))  # 注意 center 是 (y, x)
    #     compact = generate_compact_mask(center, mask, area_ratio=1.0)  # area_ratio=1.0 保持原面积
    #     compact_masks.append(compact)

    # 也可以用whole cluster masks生成compact mask，面积可以取1/4
    for idx, mask in enumerate(whole_cluster_masks):
        center = tuple(map(int, centers[idx]))  # 注意 center 是 (y, x)
        compact = generate_compact_mask(center, mask, area_ratio=4.0)
        compact_masks.append(compact)

    # ------------------------- vis start -------------------------
    h, w = compact_masks[0].shape
    color_mask = np.zeros((h, w, 3), dtype=np.float32)

    cmap_list = list(mcolors.TABLEAU_COLORS.values()) + list(
        mcolors.CSS4_COLORS.values()
    )
    # 给每个 compact mask 上色
    for idx, mask in enumerate(compact_masks):
        color = mcolors.to_rgb(cmap_list[idx % len(cmap_list)])
        for i in range(3):  # R, G, B 通道
            color_mask[..., i] += mask * color[i]

    # 为防止重叠位置叠加值 > 1，做 clip
    color_mask = np.clip(color_mask, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(color_mask)
    for center in centers:
        y, x = center
        plt.plot(x, y, "rx", markersize=8, label="Center")
    plt.title("Compact Cluster Masks (Color Coded)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_compact_masks.png"), dpi=300)
    plt.axis("off")
    plt.close()
    # ------------------------- vis end -------------------------

    # 创建 std mask
    mask = np.full_like(processed_map, fill_value=alpha_background, dtype=np.float32)
    # 合并所有 compact_mask 为一个总的 bool mask
    combined_mask = np.any(compact_masks, axis=0)  # shape: same as each compact_mask
    # 在对应位置赋值
    mask[combined_mask] = alpha_content

    # final： 应用 mask 到 latents

    from scipy.ndimage import zoom
    mask = zoom(mask, (128 / 32, 128 / 32), order=1)  # order=1 为双线性插值

    mask = torch.from_numpy(mask).to(new_latents.device, dtype=new_latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度

    mask_bool = mask > 1.0
    mask_bool = mask_bool.squeeze(0).squeeze(0)  # 去掉多余的维度
    masked_latents = new_latents[:, :, mask_bool]
    mean_in_mask = masked_latents.mean()
    var_in_mask = masked_latents.var()
    std_in_mask = masked_latents.std()
    print(f"Mask区域内 - 均值: {mean_in_mask:.4f}, 方差: {var_in_mask:.4f}, 标准差: {std_in_mask:.4f}")

    new_latents = new_latents * mask

    masked_latents = new_latents[:, :, mask_bool]
    mean_in_mask = masked_latents.mean()
    var_in_mask = masked_latents.var()
    std_in_mask = masked_latents.std()
    print(f"Mask区域内 - 均值: {mean_in_mask:.4f}, 方差: {var_in_mask:.4f}, 标准差: {std_in_mask:.4f}")

    return new_latents, whole_cluster_masks, compact_masks


import torch
import numpy as np


def merge_tokens_to_eos(ca_maps, threshold=0.6, mode="add"):
    """
    把前 n 个 token 的注意力 map 聚合到最后的结束符 token 上
    ca_maps: torch.Tensor, shape = [n+1, H, W]
    threshold: float, 用于生成 mask
    mode: "add" 累加, "replace" 覆盖
    """
    n_plus_1 = len(ca_maps)
    n = n_plus_1 - 1
    eos_map = ca_maps[-1].clone()  # 拿到结束符 token 的 attention map

    for i in range(n):
        token_map = ca_maps[i]
        # 得到 mask
        mask = token_map > threshold
        eos_map[mask] = token_map[mask]

    return eos_map


def ca_maps_to_masks_and_save(ca_maps_tensor, save_dir="token_masks", cmap="viridis"):
    """
    输入:
        ca_maps_tensor: [num_tokens, H, W] 的 torch.Tensor
        save_dir: 保存每个 token mask 的文件夹
        cmap: 可视化使用的 colormap
    输出:
        token_masks: [num_tokens, H, W] 的硬 mask (0/1)
    """
    os.makedirs(save_dir, exist_ok=True)

    num_tokens, H, W = ca_maps_tensor.shape

    # --- 硬分配，每个位置归属于最大 token ---
    token_id_map = ca_maps_tensor.argmax(dim=0)  # [H, W], 每个位置对应 token idx
    token_masks = torch.nn.functional.one_hot(token_id_map, num_classes=num_tokens)  # [H, W, num_tokens]
    token_masks = token_masks.permute(2, 0, 1).float()  # [num_tokens, H, W]

    # --- 可视化每个 token mask ---
    for i in range(num_tokens):
        mask = token_masks[i]
        plt.figure(figsize=(4, 4))
        plt.imshow(mask.cpu().numpy(), cmap=cmap)
        plt.title(f"Token {i} mask")
        plt.axis("off")
        save_path = os.path.join(save_dir, f"token_{i}_mask.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    print(f"保存完成，路径: {os.path.abspath(save_dir)}")

    return token_masks


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

ALPHA_BACKGROUD = 0.975
ALPHA_CONTENT = 1.1

if __name__ == "__main__":
    # ==================================================
    # 1. init model pipeline
    # ==================================================
    model_id = PIPELINE_PATH
    pipe = AttentionGuidanceStableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe.scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    # ==================================================
    #  3. prepare prompt
    # ==================================================
    json_path = "/home/hdu/lxl/Break-Noise-Bias/gpt.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    seeds = random.sample(range(1, 2**31 - 1), 5)
    for i, item in enumerate(data):
        prompt = item["prompt"]
        obj = item["objects"]
        num_int = item["counts"]
        indices_to_alter = item["indices_to_alter"]
        token_counts = item["token_counts"]
        scounts = ""
        for c in token_counts.values():
            scounts += str(c)
        seeds = [314455853]
        print("seeds", seeds)
        for seed in seeds:
            SAVE_DIR = f"output_multi/{seed}/ours"
            SAVE_DIR_ORIGINAL = f"output_multi/{seed}/original"
            SAVE_DIR_VIS = f"output_multi/{seed}/vis"
            os.makedirs(SAVE_DIR_VIS, exist_ok=True)
            os.makedirs(SAVE_DIR, exist_ok=True)
            os.makedirs(SAVE_DIR_ORIGINAL, exist_ok=True)

            print(f"{i} | {prompt} | {obj} | {token_counts.values()} | {seed}")
            generator = set_seed(seed)
            latents = torch.randn(
                (1, 4, 128, 128), generator=generator, device=DEVICE, dtype=torch.float16
            )
            new_generator = torch.Generator(device=DEVICE)
            new_generator.set_state(generator.get_state())
            # ==================================================
            #  5. call model for original generation
            # ==================================================
            controller = AttentionStore()
            ptp_utils.register_attention_control(model=pipe, controller=controller)
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=50,
                    latents=latents,
                    generator=generator,
                    token_index=indices_to_alter,
                    token_counts=token_counts
                ).images[0]

            aggregated_attention_store_per_step = vis_utils.show_cross_attention_per_step(
                attention_store=controller,
                prompt=prompt,
                tokenizer=pipe.tokenizer,
                res=32,
                indices_to_alter=indices_to_alter,
            )
            ca_output_dir = os.path.join(SAVE_DIR_VIS, f"{obj}_{scounts}_original_CA")

            os.makedirs(ca_output_dir, exist_ok=True)
            for step in aggregated_attention_store_per_step:
                for word in aggregated_attention_store_per_step[step]:
                    ca_map = aggregated_attention_store_per_step[step][word]
                    # 转换为 numpy 数组（如果是 torch 张量）
                    if torch.is_tensor(ca_map):
                        ca_map = ca_map.detach().cpu().numpy()

                    plt.figure(figsize=(6, 6))
                    plt.imshow(ca_map, cmap='viridis')  # 你也可以用 'hot', 'jet', 'plasma' 等 colormap
                    # plt.colorbar()
                    # plt.title("Cross Attention Map")
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(os.path.join(ca_output_dir, f"{prompt}_step{step}_{word}.png"),
                                bbox_inches='tight')
                    plt.close()

            output_path = os.path.join(SAVE_DIR_ORIGINAL, f"{i}_{obj}_{scounts}.png")
            image.save(
                output_path
            )
            del controller
            del image

            # ==================================================
            #  6. recombine noise
            # ==================================================
            step_for_read_CA_list = range(1, 21, 1)  # 1 到 20 步
            words = list(aggregated_attention_store_per_step[0].keys())
            ca_maps = []  # 用 list 存放每个 word 的平均 map
            for word in words:
                per_word_maps = []
                for step_for_read_CA in step_for_read_CA_list:
                    word_ca_map_dict = aggregated_attention_store_per_step[step_for_read_CA - 1]
                    ca_map_tmp = word_ca_map_dict[word]  # 每一步该词的 CA map
                    per_word_maps.append(ca_map_tmp)
                # 平均得到该词的最终 map
                ca_map = torch.stack(per_word_maps).mean(dim=0)
                processed_map = process_attention_map(ca_map, steps=0)  # 归一化
                ca_maps.append(processed_map)

            # ca_map = ca_maps[-1]
            ca_maps_tensor = torch.stack(ca_maps, dim=0)  # [num_words, H, W]
            ca_map = ca_maps_tensor.max(dim=0).values

            # 生成 mask 并保存
            # token_masks = ca_maps_to_masks_and_save(ca_maps_tensor, save_dir="token_masks_example")

            latents, whole_cluster_masks, compact_masks = noise_recreate(
                latents,
                ca_map,
                numeral=num_int,
                alpha_background=ALPHA_BACKGROUD,
                alpha_content=ALPHA_CONTENT,
                save_dir=f"{SAVE_DIR_VIS}/vis",
                prefix=f"{obj}_{scounts}",
            )

            # ==================================================
            #  7. call model for second generation
            # ==================================================

            pipe = AttentionGuidanceStableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )
            pipe.scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(DEVICE)
            perform_guidance_flag = True
            controller = LossGuideAttentionStore(
                attn_res=32, loss=perform_guidance_flag
            )
            ptp_utils.register_attention_control(model=pipe, controller=controller,
                                                 perfrom_loss_guidance=perform_guidance_flag)
            print(generator.get_state())

            ca_output_dir = os.path.join(
                SAVE_DIR_VIS, f"{obj}_{scounts}_recreate_CA"
            )
            os.makedirs(ca_output_dir, exist_ok=True)
            sfmx_temperature = 5
            image = pipe(
                prompt,
                num_inference_steps=50,
                attention_store=controller,  # 记得提供attention store，用于为attention guidance提供mask参考
                latents=latents,
                generator=generator,
                perform_guidance_flag=perform_guidance_flag,  # 开启 attention guidance
                output_dir=ca_output_dir,  # 保存此次pipeline的CA图
                whole_cluster_masks=whole_cluster_masks,  # 用于为attention guidance提供mask参考
                softmax_temperature=sfmx_temperature,  # softmax temperature
                token_index=indices_to_alter,
                token_counts=token_counts
            ).images[0]

            output_path = os.path.join(SAVE_DIR, f"{i}_{obj}_{scounts}.png")
            image.save(output_path)

            del controller
            del image
            del latents
