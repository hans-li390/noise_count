import abc
from typing import Callable, List, Optional, Tuple, Union
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
from torchvision import transforms as T
from collections import defaultdict

from diffusers.models.attention import Attention


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
) -> Image.Image:
    """Displays a list of images in a grid."""
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


class MyAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attnstore, place_in_unet):
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # # Store Attention
        # self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."

        # residual = hidden_states
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        # input_ndim = hidden_states.ndim

        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query = attn.to_q(hidden_states)

        # if encoder_hidden_states is None:
        #     encoder_hidden_states = hidden_states
        # elif attn.norm_cross:
        #     encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # key = attn.to_k(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)

        # inner_dim = key.shape[-1]
        # head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # if attn.norm_q is not None:
        #     query = attn.norm_q(query)
        # if attn.norm_k is not None:
        #     key = attn.norm_k(key)

        # # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        # # linear proj
        # hidden_states = attn.to_out[0](hidden_states)
        # # dropout
        # hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual

        # hidden_states = hidden_states / attn.rescale_output_factor

        # return hidden_states


class LossGuideProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store Attention
        self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(
            hidden_states,
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(
            encoder_hidden_states,
        )
        value = attn.to_v(
            encoder_hidden_states,
        )

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(
            attention_probs, is_cross, self.place_in_unet, attn.heads
        )  # store the attention maps

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](
            hidden_states,
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedSelfAttnProcessor:
    def __init__(
        self,
        attnstore,
        place_in_unet,
        whole_cluster_masks=None,
        inter_cluster_weight=1.0,
        intra_cluster_weight=1.0,
    ):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.whole_cluster_masks = whole_cluster_masks
        self.inter_cluster_weight = inter_cluster_weight  # cluster间的attention权重
        self.intra_cluster_weight = intra_cluster_weight  # cluster间的attention权重

    def set_masks(self, whole_cluster_masks):
        """设置cluster masks"""
        self.whole_cluster_masks = whole_cluster_masks

    def resize_masks_to_resolution(self, target_height, target_width):
        """将mask调整到目标分辨率"""
        if self.whole_cluster_masks is None:
            return None

        resized_masks = []
        for mask in self.whole_cluster_masks:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask.astype(np.float32))
            else:
                mask_tensor = mask.float()

            # 添加batch和channel维度用于interpolate
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # 调整大小
            resized_mask = (
                torch.nn.functional.interpolate(
                    mask_tensor, size=(target_height, target_width), mode="nearest"
                )
                .squeeze()
                .bool()
            )  # 移除额外维度并转为bool

            resized_masks.append(resized_mask)

        return resized_masks

    '''
    def create_attention_mask(self, height, width):
        """根据cluster masks创建attention mask"""
        if self.whole_cluster_masks is None:
            return None
        
        # 将原始masks调整到当前层的分辨率
        resized_masks = self.resize_masks_to_resolution(height, width)
        if resized_masks is None:
            return None
            
        # 创建attention mask: (h*w, h*w)
        num_pixels = height * width
        attention_mask = torch.zeros((num_pixels, num_pixels), dtype=torch.bool)
        
        # 对每个cluster，允许内部attention
        for mask in resized_masks:
            # 将2D mask展平为1D indices
            mask_flat = mask.flatten()
            indices = torch.where(mask_flat)[0]
            
            # 在attention mask中标记cluster内部的交互为True
            # 使用更高效的索引方式
            if len(indices) > 0:
                indices_grid = torch.meshgrid(indices, indices, indexing='ij')
                attention_mask[indices_grid[0], indices_grid[1]] = True
        
        return attention_mask
    '''

    def create_attention_mask(self, height, width):
        """根据cluster masks创建attention mask"""
        if self.whole_cluster_masks is None:
            return None

        # 将原始masks调整到当前层的分辨率
        resized_masks = self.resize_masks_to_resolution(height, width)
        if resized_masks is None:
            return None

        # 创建attention mask: (h*w, h*w)，默认全为True
        num_pixels = height * width
        attention_mask = torch.ones((num_pixels, num_pixels), dtype=torch.bool)

        # 对每个cluster独立处理
        for mask in resized_masks:
            mask_flat = mask.flatten()
            cluster_indices = torch.where(mask_flat)[0]  # 当前cluster内部的像素索引
            external_indices = torch.where(~mask_flat)[0]  # 当前cluster外部的像素索引

            if len(cluster_indices) > 0 and len(external_indices) > 0:
                # 设置 cluster内部 → cluster外部 为False
                # 使用广播的方式设置
                cluster_grid = cluster_indices.unsqueeze(1)  # (n_cluster, 1)
                external_grid = external_indices.unsqueeze(0)  # (1, n_external)

                # cluster内部到外部设为False
                attention_mask[cluster_grid, external_grid] = False

        return attention_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        else:
            # 从hidden_states推断height和width
            batch_size, sequence_length, _ = hidden_states.shape
            height = width = int(np.sqrt(sequence_length))

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # 准备原始attention mask
        original_attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 注意力分数计算
        attention_probs = attn.get_attention_scores(query, key, original_attention_mask)

        # 只对self attention应用cluster mask
        if not is_cross and self.whole_cluster_masks is not None:
            # 根据当前层的分辨率创建cluster mask
            cluster_attention_mask = self.create_attention_mask(height, width)
            if cluster_attention_mask is not None:
                cluster_attention_mask = cluster_attention_mask.to(
                    attention_probs.device
                )

                # 将mask应用到所有batch和head
                # attention_probs shape: (batch_size * num_heads, seq_len, seq_len)
                batch_heads = attention_probs.shape[0]

                # 扩展mask到所有batch和head
                expanded_mask = cluster_attention_mask.unsqueeze(0).expand(
                    batch_heads, -1, -1
                )

                # # 将mask外的attention设为（0）
                # attention_probs = attention_probs.masked_fill(
                #     ~expanded_mask, float(0)
                # )
                # cluster内部保持原值，cluster间乘以衰减因子
                attention_probs = torch.where(
                    expanded_mask,
                    attention_probs * self.intra_cluster_weight,  # cluster内增强
                    attention_probs * self.inter_cluster_weight,  # cluster间衰减
                )

        # Store attention if needed
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_attention_control(model, controller, perfrom_loss_guidance=False):
    attn_greenlist = [
        # "up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor",
        # "up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor",
        # "up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor",
        # "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
        # "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
        # "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
    ]
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        # print(name)
        # if name not in attn_greenlist:
        #     attn_procs[name] = model.unet.attn_processors[name]
        #     continue

        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        place_in_unet = ".".join(name.split(".")[:-4])
        # print(place_in_unet, name)

        cross_att_count += 1
        if perfrom_loss_guidance:  # 如果需要loss引导，则用LossGuideProcessor
            attn_procs[name] = LossGuideProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )
        else:
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def register_masked_attention_control(
    model,
    controller,
    whole_cluster_masks=None,
    inter_cluster_weight=1.0,
    intra_cluster_weight=1.0,
):
    """注册带mask的attention control"""
    attn_procs = {}
    cross_att_count = 0

    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        place_in_unet = ".".join(name.split(".")[:-4])

        # 检查是否是self attention
        is_self_attention = name.endswith("attn1.processor")

        if is_self_attention:
            # 对self attention使用masked processor
            attn_procs[name] = MaskedSelfAttnProcessor(
                attnstore=controller,
                place_in_unet=place_in_unet,
                whole_cluster_masks=whole_cluster_masks,
                inter_cluster_weight=inter_cluster_weight,
                intra_cluster_weight=intra_cluster_weight,
            )
        else:
            # 对cross attention使用原始processor
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )

        cross_att_count += 1

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


# def register_attention_control(model, controller):
#     attn_greenlist = ["up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor",
#                     "up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor",
#                     "up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor",]
#     attn_procs = {}
#     cross_att_count = 0
#     for name in model.unet.attn_processors.keys():
#         if name not in attn_greenlist:
#             attn_procs[name] = model.unet.attn_processors[name]
#             continue
#         if name.startswith("mid_block"):
#             place_in_unet = "mid"
#         elif name.startswith("up_blocks"):
#             place_in_unet = "up"
#         elif name.startswith("down_blocks"):
#             place_in_unet = "down"
#         else:
#             continue

#         cross_att_count += 1
#         attn_procs[name] = AttendExciteCrossAttnProcessor(
#             attnstore=controller, place_in_unet=place_in_unet
#         )

#     model.unet.set_attn_processor(attn_procs)
#     controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet, attn_heads)
        self.cur_att_layer += 1
        if (
            self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers
        ):  # end of step
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


# class AttentionStore(AttentionControl):

#     @staticmethod
#     def get_empty_store():
#         # return {"down_cross": [], "mid_cross": [], "up_cross": [],
#         #         "down_self": [], "mid_self": [], "up_self": []}
#         keys = {
#             "up_blocks.1.attentions.1",
#             "down_blocks.1.attentions.1",
#             "up_blocks.0.attentions.2",
#             "down_blocks.2.attentions.1",
#             "down_blocks.2.attentions.0",
#             "up_blocks.1.attentions.0",
#             "up_blocks.1.attentions.2",
#             "mid_block.attentions.0",
#             "up_blocks.0.attentions.1",
#             "down_blocks.1.attentions.0",
#             "up_blocks.0.attentions.0",
#         }
#         t = dict()
#         for key in keys:
#             t[f"{key}_self"] = []
#             t[f"{key}_cross"] = []
#         return t

#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         # print(key, attn.shape)
#         if attn.shape[1] <= 32**2:  # avoid memory overhead
#             self.step_store[key].append(attn)
#         return attn

#     def between_steps(self):
#         self.attention_store = self.step_store
#         if self.save_global_store:
#             with torch.no_grad():
#                 if len(self.global_store) == 0:
#                     self.global_store = self.step_store
#                 else:
#                     for key in self.global_store:
#                         for i in range(len(self.global_store[key])):
#                             self.global_store[key][i] += self.step_store[key][
#                                 i
#                             ].detach()
#         self.step_store = self.get_empty_store()

#     def get_average_attention(self):
#         average_attention = self.attention_store
#         return average_attention

#     def get_average_global_attention(self):
#         average_attention = {
#             key: [item / self.cur_step for item in self.global_store[key]]
#             for key in self.attention_store
#         }
#         return average_attention

#     def reset(self):
#         super(AttentionStore, self).reset()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}
#         self.global_store = {}

#     def __init__(self, save_global_store=False):
#         """
#         Initialize an empty AttentionStore
#         :param step_index: used to visualize only a specific step in the diffusion process
#         """
#         super(AttentionStore, self).__init__()
#         self.save_global_store = save_global_store
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}
#         self.global_store = {}
#         self.curr_step_index = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [], "mid_self": [], "up_self": []}
        keys = {
            "up_blocks.1.attentions.1",
            "down_blocks.1.attentions.1",
            "up_blocks.0.attentions.2",
            "down_blocks.2.attentions.1",
            "down_blocks.2.attentions.0",
            "up_blocks.1.attentions.0",
            "up_blocks.1.attentions.2",
            "mid_block.attentions.0",
            "up_blocks.0.attentions.1",
            "down_blocks.1.attentions.0",
            "up_blocks.0.attentions.0",
        }
        t = dict()
        for key in keys:
            t[f"{key}_self"] = []
            t[f"{key}_cross"] = []
        return t

    def forward(self, attn, is_cross: bool, place_in_unet: str, attn_heads=None):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # return attn
        # print(key, attn.shape)
        if attn.shape[1] <= 32**2 and key in self.target_keys:  # avoid memory overhead
            # if cross
            # if not self.save_only_curr_step:
            #     attn = attn[attn.size(0) // 2 :]
            # attn = attn.reshape(
            #     [
            #         attn.shape[0] // attn_heads,
            #         attn_heads,
            #         *attn.shape[1:],
            #     ]
            # ).mean(dim=1)
            guided_attn = attn.mean(dim=0, keepdim=True)
            self.step_store[key].append(guided_attn)

        return attn

    def between_steps(self):
        # self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][
                                i
                            ].detach()
        if self.save_store_per_step and not self.save_only_curr_step:
            # 保存每一步的attention到 attention_store_per_step
            target_key = "down_blocks.2.attentions.1_cross"
            if target_key in self.step_store and len(self.step_store[target_key]) > 0:
                # 如果只有一个张量，取第一个；如果有多个，保存整个列表
                if len(self.step_store[target_key]) == 1:
                    self.attention_store_per_step[self.cur_step] = self.step_store[
                        target_key
                    ][0].detach()
                else:
                    self.attention_store_per_step[self.cur_step] = [
                        item.detach() for item in self.step_store[target_key]
                    ]
            else:
                # 如果该层不存在或为空，保存None或空列表
                self.attention_store_per_step[self.cur_step] = None

        if self.save_only_curr_step:
            # 只保存当前步骤的attention到 attention_store_per_step (直接放到第1步)
            target_key = "down_blocks.2.attentions.1_cross"
            if target_key in self.step_store and len(self.step_store[target_key]) > 0:
                # 如果只有一个张量，取第一个；如果有多个，保存整个列表
                if len(self.step_store[target_key]) == 1:
                    self.attention_store_per_step[1] = self.step_store[target_key][0]
                else:
                    self.attention_store_per_step[1] = [
                        item for item in self.step_store[target_key]
                    ]
            else:
                # 如果该层不存在或为空，保存None或空列表
                self.attention_store_per_step[self.cur_step] = None

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_attention_per_step(self):
        return self.attention_store_per_step

    def get_average_global_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.global_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(
        self,
        save_global_store=False,
        save_store_per_step=True,
        save_only_curr_step=False,
    ):
        """
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        """
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.save_store_per_step = save_store_per_step
        self.save_only_curr_step = save_only_curr_step
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.attention_store_per_step = {}
        self.global_store = {}
        self.target_keys = ["down_blocks.2.attentions.1_cross"]
        # self.curr_step_index = 0


class LossGuideAttentionStore:
    @staticmethod
    def get_empty_step_store(save_timesteps=None):
        d = defaultdict(list)
        for t in save_timesteps:
            d[t] = {}
        return d

    @staticmethod
    def get_empty_store():
        keys = {
            "up_blocks.1.attentions.1",
            "down_blocks.1.attentions.1",
            "up_blocks.0.attentions.2",
            "down_blocks.2.attentions.1",
            "down_blocks.2.attentions.0",
            "up_blocks.1.attentions.0",
            "up_blocks.1.attentions.2",
            "mid_block.attentions.0",
            "up_blocks.0.attentions.1",
            "down_blocks.1.attentions.0",
            "up_blocks.0.attentions.0",
        }
        t = dict()
        for key in keys:
            t[f"{key}_self"] = []
            t[f"{key}_cross"] = []
        return t
        # return {}

    def __init__(
        self,
        attn_res,
        loss,
    ):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """

        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.curr_step_index = 0
        self.count_ts = 0

        self.attn_res = (attn_res, attn_res)
        self.loss = loss

        # Attention store to save attention ONLY for the current step,
        self.cross_attention_store = self.get_empty_store()
        self.self_attention_store = self.get_empty_store()

        self.all_cross_attention = {}
        self.all_self_attention = {}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if (attn.shape[1] == np.prod(self.attn_res)) and (self.cur_att_layer >= 0):
            if (not self.loss) and (is_cross):
                guided_attn = attn[attn.size(0) // 2 :]
                print(f"guided_attn shape: {guided_attn.shape}")
            else:
                guided_attn = attn
            if is_cross:
                guided_attn = guided_attn.reshape(
                    [
                        guided_attn.shape[0] // attn_heads,
                        attn_heads,
                        *guided_attn.shape[1:],
                    ]
                ).mean(dim=1)
                # self.cross_attention_store[place_in_unet] = guided_attn
                self.cross_attention_store[key].append(guided_attn)

            else:
                pass

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.all_cross_attention = self.cross_attention_store
        self.cross_attention_store = self.get_empty_store()
        self.all_self_attention = self.self_attention_store
        self.self_attention_store = self.get_empty_store()

    def aggregate_attention(
        self, from_where: List[str], get_cross=True
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        if get_cross:
            attention_maps = self.all_cross_attention
        else:
            attention_maps = self.all_self_attention

        for layer, curr_map in attention_maps.items():
            if curr_map == []:
                continue
            if any([x in layer for x in from_where]):
                curr_map = torch.stack(curr_map, dim=0)  # shape: [10, 1, 1024, 77]
                curr_map = curr_map.mean(dim=0)  # shape: [1, 1024, 77]
                curr_map_reshape = curr_map.reshape(
                    -1, self.attn_res[0], self.attn_res[1], curr_map.shape[-1]
                )
                out.append(curr_map_reshape)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out


def aggregate_attention(
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
) -> torch.Tensor:
    """Aggregates the attention across the different layers and heads at the specified resolution."""
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    if out != []:
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
    return out


def cal_threshold(img):
    """
    img: 1*h*w
    """
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    # 先进行高斯滤波，再使用Otsu阈值法
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = blur.astype("uint8")
    ret3, th3 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def get_crossMask(
    attention_store: AttentionStore,
    res: int,
    from_where: List[str],
    select: int,
    indices: List[int],
    save_name: str,
) -> torch.Tensor:
    """get cross-attn mask"""

    toImg = T.ToPILImage()
    corss_attn_map = aggregate_attention(
        attention_store, res, from_where, True, select
    )  # h w 77

    h, w, seq_len = corss_attn_map.shape
    corss_attn_map = corss_attn_map.permute(2, 0, 1)  # 77 h w
    amap_glo = np.zeros((h, w))

    for index in indices:
        attn_map = corss_attn_map[index]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        attn_map = attn_map**2
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        attn_map = attn_map**2
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        amap = attn_map.reshape(1, 32, 32).float()
        # amap = amap/amap.sum()
        amap = cal_threshold(amap)
        amap_glo = amap_glo + amap.reshape(32, 32)

    mask = amap_glo
    mask = torch.from_numpy(mask)
    toImg(mask.reshape(1, h, w)).save(save_name)
    mask = mask != 0  # bool tensor
    attention_store.mask = mask


def attn_refine(amap, coor_index=None):
    """
    amap: batch_size*head_nums h*w h*w
    coor_index: (h*w), bool tensor
    """
    bh, tnum, res = amap.shape
    grid_size = h = w = int(torch.sqrt(torch.tensor(res)))

    amap = amap[:, coor_index, :]  # bh tnum res
    bh, tnum, res = amap.shape

    x = torch.arange(grid_size).float()
    y = torch.arange(grid_size).float()
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_xy = torch.stack((grid_x, grid_y), dim=2).cuda()  # shape: h*w*2
    amap = amap.reshape(bh, tnum, h, w)

    # mu = torch.einsum('bijk,jkl->bil', amap, grid_xy) #shape: bh tnum 2
    # mu = mu.reshape(bh,tnum,1,1,2)

    mu = grid_xy.reshape(-1, 2)[coor_index, :].reshape(1, tnum, 1, 1, 2)
    # mu = torch.tensor([12.,20.]).reshape(1,1,1,1,1,2)
    mu = mu.repeat(bh, 1, 1, 1, 1)  # bh tmnm 1 1 2
    xy_norm = grid_xy.view(1, 1, h, w, 2) - mu  # shape: bh tnum h w 2

    xy_norm = xy_norm.reshape(-1, 2, 1)
    xy_square = torch.bmm(xy_norm, xy_norm.permute(0, 2, 1)).reshape(
        bh, tnum, h, w, 2, 2
    )  # bh, tnum, h, w,2,2

    sigma = torch.einsum("bijk,bijklm->bilm", amap, xy_square)  # bh tnum 2 2
    # sigma[:,:] = torch.tensor([[1,0.],[0,1.]]).cuda()

    inv_sigma = torch.linalg.inv(sigma)  # bh tnum 2 2
    inv_sigma = inv_sigma.reshape(bh, tnum, 1, 1, 2, 2)
    inv_sigma = inv_sigma.repeat(1, 1, h, w, 1, 1)  # bh tnum h w 2 2

    dis = torch.bmm(xy_norm.permute(0, 2, 1), inv_sigma.reshape(-1, 2, 2))  # -1 1 2
    dis = torch.bmm(dis, xy_norm).reshape(bh, tnum, h, w)  # bh, tnum, h, w

    dis = torch.sqrt(dis)  # bh tnum h w

    dis2 = (dis - dis.amin(dim=(-2, -1), keepdim=True)) / (
        dis.amax(dim=(-2, -1), keepdim=True) - dis.amin(dim=(-2, -1), keepdim=True)
    )
    dis2 = torch.exp(-dis2 / 0.5)  # bh tnum h w

    amap = amap * dis2
    amap = (amap - amap.amin(dim=(-2, -1), keepdim=True)) / (
        amap.amax(dim=(-2, -1), keepdim=True) - amap.amin(dim=(-2, -1), keepdim=True)
    )
    # amap = amap**1.5

    amap = amap / amap.sum(dim=(-2, -1), keepdim=True)

    return amap.reshape(bh, tnum, h * w)


class CompactAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if attn.eot is not None:
            value2 = attn.to_v(attn.eot, *args)
            # print(value2.shape, value.shape)
            value[1][8:] = value2[0][8:]

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_self_time(pipe, i):
    for name, module in pipe.unet.named_modules():
        # if name in attn_greenlist:
        if (name.startswith("mid_block")) and name.endswith("attn1"):
            # if name.endswith("attn1"):
            # if name.startswith("down_blocks.2") and name.endswith("attn1"):
            setattr(module, "time", i)


def register_self_eot(pipe, eot):
    for name, module in pipe.unet.named_modules():
        setattr(module, "eot", eot)
