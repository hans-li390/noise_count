import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd

def show_cross_attention_per_step(
    prompt: str,
    attention_store: AttentionStore,
    tokenizer,
    indices_to_alter: List[int],
    res: int,
    # from_where: List[str],
    select: int = 0,
    # orig_image=None,
    change2cpu = True,
):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    # aggregate attention maps
    aggregated_attention_store_per_step = {}
    attention_store_per_step = attention_store.get_attention_per_step()
    num_pixels = res**2
    for step in range(len(attention_store_per_step)):
        aggregated_attention_store_per_step[step] = {}
        out = []
        for item in attention_store_per_step[step+1]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
        if out != []:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
        attention_maps = out
        if change2cpu:
            attention_maps = attention_maps.detach().cpu()
        for i in range(len(tokens)):
            if i in indices_to_alter:
                attention_map = attention_maps[:, :, i]
                word =  decoder(int(tokens[i]))
                aggregated_attention_store_per_step[step][word] = attention_map
    return aggregated_attention_store_per_step

    
def show_cross_attention(
    prompt: str,
    attention_store: AttentionStore,
    tokenizer,
    indices_to_alter: List[int],
    res: int,
    from_where: List[str],
    select: int = 0,
    orig_image=None,
):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    if attention_maps == []:
        return False
    attention_maps = attention_maps.detach().cpu()
    images = []
    image = np.array(orig_image.resize((res**2, res**2)))
    image = ptp_utils.text_under_image(image, "original")
    images.append(image)
    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res**2, res**2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    return ptp_utils.view_images(np.stack(images, axis=0))


def save_attention(
    prompt: str,
    attention_store: AttentionStore,
    tokenizer,
    indices_to_alter: List[int],
    res: int,
    from_where: List[str],
    select: int = 0,
    save_path: str = "",
):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    if attention_maps == []:
        return False
    attention_maps = attention_maps.detach().cpu()

    for i in range(len(tokens)):
        attention_map = attention_maps[:, :, i]
        attention_map = attention_maps[:, :, i].unsqueeze(0).unsqueeze(0)  # [1,1,32,32]
        attention_map = attention_map.float()
        # 使用双线性插值放大到128x128
        attention_map = F.interpolate(
            attention_map, size=128, mode="bilinear", align_corners=False
        )
        attention_map = attention_map.squeeze()  # 变回[128,128]

        if i in indices_to_alter:
            token_name = decoder(int(tokens[i]))
            # save_path_pt = save_path + f"_{token_name}.pt"
            # torch.save(attention_map, save_path_pt)
            save_path_npy = save_path + f".npy"
            np.save(save_path_npy, attention_map.numpy())
            # save_path_csv = save_path + f"_{token_name}.csv"
            # pd.DataFrame(attention_map.numpy()).to_csv(save_path_csv, index=False)
            save_path_png = save_path + f".png"
            plt.imsave(save_path_png, attention_map.numpy(), cmap="jet")


def read_attention(path):
    if path.endswith(".pt"):
        return torch.load(path)
    elif path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError("path must end with .pt or .npy")


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res**2, relevnace_res**2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(
        1, 1, image_relevance.shape[-1], image_relevance.shape[-1]
    )
    image_relevance = (
        image_relevance.cuda(1)
    )  # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=relevnace_res**2, mode="bilinear"
    )
    image_relevance = image_relevance.cpu()  # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    image_relevance = image_relevance.reshape(relevnace_res**2, relevnace_res**2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    #  # 使用 matplotlib 创建图像和 colorbar，并保存为 np.array
    # fig, ax = plt.subplots(figsize=(4, 4))
    # im = ax.imshow(vis)
    # ax.axis('off')
    # cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    # cbar.set_label('Relevance (Attention Score)', rotation=270, labelpad=15)

    # # 把整个 figure 渲染成图像
    # canvas = FigureCanvas(fig)
    # canvas.draw()
    # image_with_legend = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # image_with_legend = image_with_legend.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.close(fig)
    # return image_with_legend
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new("RGB", (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
