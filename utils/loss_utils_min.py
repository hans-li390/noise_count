import torch
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.nn import functional as F
try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

EPS = 1e-12


def ensure_region_masks_list(region_masks):
    if isinstance(region_masks, list):
        return [torch.as_tensor(r, dtype=torch.float32) for r in region_masks]
    if isinstance(region_masks, torch.Tensor) and region_masks.dim() == 3:
        return [region_masks[r].float() for r in range(region_masks.shape[0])]
    raise ValueError("region_masks must be list or tensor [R,H,W]")


def match_regions_to_tokens_with_counts_minmax(ca_maps_tensor, region_masks_list, target_counts):
    """
    ca_maps_tensor: torch.Tensor [K, H, W] (min-max normalized per-token)
    region_masks_list: list of R masks [H,W]
    target_counts: dict {token_idx: desired_count}  (sum_counts ideally == R)
    Returns:
       region_to_token: dict {region_idx: token_idx}
       token_to_regions: dict {token_idx: [region_idx,...]}
       scores: torch.Tensor [K, R] (pos fractions used)
    Notes:
       - If sum_counts != R this function will still return a mapping following rules discussed below.
    """
    target_counts = {int(k): int(v) for k, v in target_counts.items()}

    device = ca_maps_tensor.device
    dtype = ca_maps_tensor.dtype
    K, H, W = ca_maps_tensor.shape
    region_masks_list = ensure_region_masks_list(region_masks_list)
    R = len(region_masks_list)

    # flatten token maps and normalize per-token mass
    maps_flat = ca_maps_tensor.reshape(K, -1)  # [K, H*W]
    total_mass = maps_flat.sum(dim=1)  # [K]
    mass_norm = torch.zeros_like(maps_flat)
    for k in range(K):
        tm = total_mass[k]
        if tm.item() > EPS:
            mass_norm[k] = maps_flat[k] / (tm + EPS)
        else:
            mass_norm[k] = torch.zeros_like(maps_flat[k])

    # region vectors (match dtype)
    region_vecs = [r.to(device=device, dtype=maps_flat.dtype).reshape(-1) for r in region_masks_list]
    region_mat = torch.stack(region_vecs, dim=1)  # [H*W, R]

    # scores: pos fractions
    scores = mass_norm @ region_mat  # [K, R], each entry ∈ [0,1]

    # build expanded rows according to target_counts
    token_rows = []
    token_row_to_token = []  # map expanded-row-index -> original token idx
    for k in range(K):
        cnt = int(target_counts.get(k, 0))
        if cnt <= 0:
            continue
        for _ in range(cnt):
            token_rows.append(scores[k].detach().cpu().numpy())  # row vector length R
            token_row_to_token.append(int(k))

    if len(token_rows) == 0:
        # no requested counts: fallback to greedy many-to-one: assign each region to argmax token
        region_to_token = {}
        token_to_regions = {k: [] for k in range(K)}
        scores_np = scores.detach().cpu().numpy()
        for r in range(R):
            best_k = int(np.argmax(scores_np[:, r]))
            region_to_token[r] = best_k
            token_to_regions[best_k].append(r)
        return region_to_token, token_to_regions, scores

    rows_np = np.stack(token_rows, axis=0)  # [S, R] where S = sum_counts
    S = rows_np.shape[0]

    # If S > R: we cannot get more matches than R. We'll still run Hungarian on SxR padded to n x n,
    # but only at most R matches will be real (cols < R). Unmatched token-instances will be considered missing.
    n = max(S, R)
    cost = np.zeros((n, n), dtype=float)
    cost[:S, :R] = -rows_np  # maximize scores
    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        region_to_token = {}
        token_to_regions = {k: [] for k in range(K)}
        for r_idx, c_idx in zip(row_ind, col_ind):
            # if both indices are in real ranges, keep
            if r_idx < S and c_idx < R:
                original_token = token_row_to_token[r_idx]
                region_to_token[c_idx] = original_token
                token_to_regions[original_token].append(c_idx)
    else:
        # fallback greedy: assign highest scoring (row, col) pairs until columns exhausted or rows exhausted
        region_to_token = {}
        token_to_regions = {k: [] for k in range(K)}
        # produce list of (row, col, score)
        triples = []
        for r_idx in range(S):
            for c_idx in range(R):
                triples.append((r_idx, c_idx, float(rows_np[r_idx, c_idx])))
        # sort desc by score
        triples.sort(key=lambda x: x[2], reverse=True)
        assigned_cols = set()
        for r_idx, c_idx, sc in triples:
            if c_idx in assigned_cols:
                continue
            original_token = token_row_to_token[r_idx]
            region_to_token[c_idx] = original_token
            token_to_regions[original_token].append(c_idx)
            assigned_cols.add(c_idx)
            if len(assigned_cols) >= R:
                break

    # Postprocess: if sum_counts < R then some regions remain unassigned -> assign greedily to best token
    assigned_regions_set = set(region_to_token.keys())
    if len(assigned_regions_set) < R:
        scores_np = scores.detach().cpu().numpy()
        for r in range(R):
            if r in assigned_regions_set:
                continue
            best_k = int(np.argmax(scores_np[:, r]))
            region_to_token[r] = best_k
            token_to_regions[best_k].append(r)

    return region_to_token, token_to_regions, scores


# ======================
#  对ca map归一化
# 过滤每个区域内的low注意力值
# 并做ca、area双重归一化
# 计算损失时不仅仅是平均，还要先softmax
def compute_loss(ca_map, region_masks, **kwargs):
    """
    ca_map: Tensor of shape (H, W)
    region_masks: list of numpy arrays or BoolTensor of shape (H, W)
    """
    temperature = kwargs.pop("temperature", 0.0)  # 3～10 越大 ⇒ 权重越集中到高损失区域（接近 max） 越小 ⇒ 趋近于平均（接近 mean）
    iteration = kwargs.pop("iteration", 99)
    refinement = kwargs.pop("refinement", -1)
    
    # ====================== CONFIG ======================
    TOPKP = 0.21 # percentage of top attention values to compute loss in each region
    # ====================== CONFIG ======================
    
    # # 高斯平滑
    # ca_map = gaussian_smooth_map(ca_map, kernel_size=5, sigma=1.0)
    
    device = ca_map.device
    H, W = ca_map.shape
    WHOLEAREA = H * W  # 整个区域的面积
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    total_loss = 0.0
    count = 0
    losses = []
    centers = []
    for region_mask in region_masks:
        # 转换为 tensor 并放到正确设备
        if isinstance(region_mask, np.ndarray):
            mask = torch.from_numpy(region_mask).to(device)
        else:
            mask = region_mask.to(device)
        mask = mask.bool()  # 确保是布尔类型
        
        # 1. 获取region mask内的注意力分布，并进行归一化处理
        region_attn_values = ca_map[mask] 
        max_val = region_attn_values.max()
        min_val = region_attn_values.min()
        norm_region_attn = (region_attn_values - min_val) / (max_val - min_val)
        attn_in_region = torch.zeros_like(ca_map)
        attn_in_region[mask] = norm_region_attn
        
        # 2. 过滤掉低注意力值
        nonzero = attn_in_region[attn_in_region > 0]
    
        k = max(1, int(nonzero.numel() * TOPKP))  # 至少保留1个
        threshold = torch.kthvalue(nonzero, nonzero.numel() - k + 1).values
        attn_in_region = torch.where(attn_in_region >= threshold, attn_in_region, torch.zeros_like(attn_in_region))
        
        # 3. 计算注意力值在区域内的总和
        sum_attn = attn_in_region.sum()
        if sum_attn == 0:
            continue  # skip empty region
        
        # 4. 计算注意力值在区域内的质心位置
        # attention weighted centroid
        x_center = (attn_in_region * x_grid).sum() / sum_attn
        y_center = (attn_in_region * y_grid).sum() / sum_attn
        centers.append((y_center.item(), x_center.item()))
        
        # 5. 计算所有点到质心的距离平方，并根据区域内的注意力值加权，得到损失
        # distance to center
        dist2 = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2  # (H, W)
        # weighted distance penalty
        area = mask.sum()
        loss = (attn_in_region * dist2).sum() / sum_attn / area * WHOLEAREA # normalize by sum of attention and mask area
        
        
        # 6. 累加损失
        total_loss += loss
        losses.append(loss)
        count += 1

    # # ----------------------
    # # 可视化 ca_map
    # # ----------------------
    # save_dir = "mask_in_guidance_denoising"
    # os.makedirs(save_dir, exist_ok=True)
    # if torch.is_tensor(ca_map):
    #     ca_map_vis = ca_map.detach().cpu().numpy()
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(ca_map_vis, cmap="viridis")
    #     plt.colorbar()
    #     plt.title("Cross Attention Map")
    #     plt.axis("off")  # 关闭坐标轴
    #     for y, x in centers:
    #         plt.scatter(x, y, c='red', s=30, marker='x')  # 红色叉标记中心
    #     plt.savefig(
    #         f"{save_dir}/{iteration}_{refinement}.png",
    #         bbox_inches="tight",
    #     )
    #     plt.close()
    # # ----------------------
    # # ----------------------
    
    # if count == 0:
    #     return torch.tensor(0.0, device=device)
    if len(losses) == 0:
        return torch.tensor(0.0, device=device), region_masks
    
    losses = torch.stack(losses)
    weights = torch.softmax(losses * temperature, dim=0) 
    # print("weights:", weights)
    final_loss = (weights * losses).sum()
    return final_loss, region_masks  # average over all regions

import torch
import numpy as np
from typing import Optional, Tuple, List

def compute_max_attention_per_region_object_maps(object_attention_maps: torch.Tensor,  # [num_tokens, H, W]
                                                 region_masks: List[torch.Tensor],  # list of [H*W] bool tensor
                                                 region_to_token: dict,  # {region_id: token_id}
                                                 use_smoothing: bool = True,
                                                 smoothing_kernel: int = 5,
                                                 area_threshold: int = 32):
    device = object_attention_maps.device
    _, H, W = object_attention_maps.shape
    max_list = []

    for region_id, token_id in region_to_token.items():
        mask = region_masks[region_id]  # boolean mask [H*W]
        if mask.ndim == 2:  # [H, W] → flatten
            mask = mask.reshape(-1)
        if mask.sum() < area_threshold:
            max_list.append(torch.tensor(0., device=device))
            continue

        # flatten attention map for this token
        attn_flat = object_attention_maps[token_id].reshape(-1)  # [H*W]
        vals = attn_flat[mask]  # select pixels inside region mask

        if use_smoothing:
            # reshape attn map
            attn_2d = object_attention_maps[token_id]  # [H, W]

            # masked map (match dtype with attn_2d)
            full_map = torch.zeros((1, 1, H, W), device=device, dtype=attn_2d.dtype)
            full_map[0, 0][mask.reshape(H, W)] = attn_2d[mask.reshape(H, W)]

            # build smoothing kernel (same dtype)
            kernel = torch.ones((1, 1, smoothing_kernel, smoothing_kernel),
                                device=device, dtype=attn_2d.dtype)
            kernel = kernel / kernel.sum()
            pad = smoothing_kernel // 2

            # conv2d smoothing
            sm = F.conv2d(full_map, kernel, padding=pad)
            max_val = sm.max()
        else:
            max_val = vals.max()

        max_list.append(max_val)

    return max_list  # list of tensors (one per region)

def compute_ae_loss(max_attention_per_token: List[torch.Tensor],
                                     target: float = 1.0,
                                     return_losses: bool = False):
    """
    原始 A&E: losses_t = max(0, target - max_attention_t); final loss = max_t losses_t
    """
    losses = [torch.relu(target - v) for v in max_attention_per_token]
    if len(losses) == 0:
        loss = torch.tensor(0., device=max_attention_per_token[0].device if max_attention_per_token else 'cpu')
    else:
        loss = torch.stack(losses).max()
    if return_losses:
        return loss, losses
    return loss


def total_loss(object_attention_map, ca_maps_tensor, whole_cluster_masks, token_counts, scale=1.0, refinement=0, iteration=0, temperature=1.0):
    """
    计算总损失：compute_loss + scale * compute_ae_loss

    Args:
        object_attention_map: torch.Tensor, 最终 attention map，用于 compute_loss
        ca_maps_tensor: torch.Tensor, 所有 token 的 attention maps, shape [num_tokens, H, W]
        whole_cluster_masks: torch.Tensor, 区域掩码
        token_counts: dict, 每个 token 出现次数
        scale: float, ae_loss 的缩放系数
        iteration: int, 当前迭代步（compute_loss 需要）
        temperature: float, softmax temperature (compute_loss 需要)

    Returns:
        total_loss: torch.Tensor, 标量总损失
        ae_loss: torch.Tensor, 标量 ae_loss
        per_token_losses: list, 每个 token 对应的 ae_loss
        whole_cluster_masks: 更新后的 cluster masks
    """
    # Step 1: 匹配 token 与区域
    region_to_token, token_to_region, _ = match_regions_to_tokens_with_counts_minmax(
        ca_maps_tensor, whole_cluster_masks, token_counts
    )
    print("token_to_region", token_to_region)
    # Step 2: 计算每个区域的最大 attention
    max_list = compute_max_attention_per_region_object_maps(ca_maps_tensor,
                                                                 whole_cluster_masks,
                                                                 region_to_token)
    # Step 3: 计算 ae_loss
    ae_loss, per_token_losses = compute_ae_loss(max_list, target=1.0, return_losses=True)
    # Step 4: 计算 compute_loss
    loss, whole_cluster_masks = compute_loss(ca_map=object_attention_map,
                                             region_masks=whole_cluster_masks,
                                             iteration=iteration,
                                             refinement=refinement,
                                             temperature=temperature)
    print("ae_loss", ae_loss.item() * scale, "var_loss", loss.item())
    # Step 5: 合并总损失
    total_loss_val = loss + scale * ae_loss

    return total_loss_val, ae_loss, per_token_losses, whole_cluster_masks
