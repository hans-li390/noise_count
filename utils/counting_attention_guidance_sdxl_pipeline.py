import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torch.fft
from utils.loss_utils import total_loss
import gc
from diffusers.image_processor import PipelineImageInput
import os
from diffusers.utils import (
    deprecate,
    logging,
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from utils import ptp_utils
from utils.ptp_utils import AttentionStore, LossGuideAttentionStore

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

# from utils.ptp_utils import AttentionStore, aggregate_attention, register_self_time
from torchvision import transforms as T

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

EPS = 1e-12



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class AttentionGuidanceStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
    ]

    def perform_iterative_refinement_step(
        self,
        loss,
        latents,
        attention_store,
        whole_cluster_masks,
        curr_prompt_embeds,
        timestep_cond,
        added_cond_kwargs,
        t,
        threshold,
        object_token_idx,
        i,
        step_size,
        token_counts,
        scale,
        max_refinement_steps=20
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = threshold

        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            _ = self.unet(
                latents,
                t,
                encoder_hidden_states=curr_prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            self.unet.zero_grad()

            cross_attention_maps = (
                attention_store.aggregate_attention(
                    from_where=["up", "mid", "down"],
                    get_cross=True,
                )
            )
            cross_attention_maps = cross_attention_maps.permute(
                2, 0, 1
            )
            # object_attention_map = cross_attention_maps[object_token_idx]
            object_attention_maps = []
            for token in object_token_idx:
                attn_map = cross_attention_maps[token]
                object_attention_map = self.process_attention_map(attn_map)
                object_attention_maps.append(object_attention_map)
            # object_attention_map = self.merge_tokens_to_eos(object_attention_maps)
            ca_maps_tensor = torch.stack(object_attention_maps, dim=0)  # [num_words, H, W]
            object_attention_map = ca_maps_tensor.max(dim=0).values
            # ACL部分，方差变小
            # loss, whole_cluster_masks = compute_loss(ca_map=object_attention_map, region_masks=whole_cluster_masks, iteration=i, refinement = iteration, temperature=self.softmax_temperature)
            loss, _, _, whole_cluster_masks = total_loss(object_attention_map=object_attention_map,
                                                         ca_maps_tensor=ca_maps_tensor,
                                                         whole_cluster_masks=whole_cluster_masks,
                                                         token_counts=token_counts,
                                                         scale=scale,
                                                         iteration=i,
                                                         refinement=iteration,
                                                         temperature=self.softmax_temperature)
            if loss != 0:
                latents = self.update_latent(
                    latents=latents,
                    loss=loss,
                    step_size=step_size,
                )

                print(f"Iteration {i} refinement {iteration} | Loss: {loss:0.4f}")

            if iteration >= max_refinement_steps:
                # print(f"Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(
            latents,
            t,
            encoder_hidden_states=curr_prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        self.unet.zero_grad()

        cross_attention_maps = (
            attention_store.aggregate_attention(
                from_where=["up", "mid", "down"],
                get_cross=True,
            )
        )
        cross_attention_maps = cross_attention_maps.permute(
            2, 0, 1
        )
        object_attention_maps = []
        for token in object_token_idx:
            attn_map = cross_attention_maps[token]
            object_attention_map = self.process_attention_map(attn_map)
            object_attention_maps.append(object_attention_map)
        # object_attention_map = self.merge_tokens_to_eos(object_attention_maps)
        ca_maps_tensor = torch.stack(object_attention_maps, dim=0)  # [num_words, H, W]
        object_attention_map = ca_maps_tensor.max(dim=0).values

        loss, _, _, whole_cluster_masks = total_loss(object_attention_map=object_attention_map,
                                                     ca_maps_tensor=ca_maps_tensor,
                                                     whole_cluster_masks=whole_cluster_masks,
                                                     token_counts=token_counts,
                                                     scale=scale,
                                                     iteration=i,
                                                     refinement= "final",
                                                     temperature=self.softmax_temperature)
        print(f"Iteration {i} refinement finished | Loss: {loss:0.4f}")
        return loss, latents

    def update_latent(self, latents, loss, step_size):
        grad = torch.autograd.grad(loss, latents, create_graph=False)[0]
        print(f"Grad : {grad.mean().item():0.4f}")
        latents = latents - step_size * grad
        return latents

    def process_attention_map(self, att_map, steps=1):
        """
        norm and square，最大最小值归一化
        """
        for _ in range(steps):
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            att_map = att_map ** 2
        # 最后再归一化一次
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        return att_map

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        attention_store: LossGuideAttentionStore = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        perform_guidance_flag: bool = False,
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        output_dir = kwargs.pop("output_dir", None)
        whole_cluster_masks = kwargs.pop("whole_cluster_masks", None)
        softmax_temperature = kwargs.pop("softmax_temperature", 0.0)
        self.softmax_temperature = softmax_temperature
        token_index = kwargs.pop("token_index")
        token_counts = kwargs.pop("token_counts")
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        # For guidance optimization
        guidance_prompt_embeds = (
            prompt_embeds[prompt_embeds.size(0) // 2 :]
            if self.do_classifier_free_guidance
            else prompt_embeds
        )
        guidance_add_text_embeds = (
            add_text_embeds[add_text_embeds.size(0) // 2 :]
            if self.do_classifier_free_guidance
            else add_text_embeds
        )
        guidance_add_time_ids = (
            add_time_ids[add_time_ids.size(0) // 2 :]
            if self.do_classifier_free_guidance
            else add_time_ids
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        torch.cuda.empty_cache()

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with (self.progress_bar(total=num_inference_steps) as progress_bar):
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # =========================================================
                # Perform Loss Optimization Beginning
                # =========================================================

                # Args preparation
                guidance_begin_step = 0  # beginning step for guidance
                guidance_end_step = 20  # end step for guidance对前20步做优化
                # perform_guidance_flag = True # whether to perform guidance or not
                if len(token_index) == 1:
                    thresholds = {
                        0: 50,
                        1: 50,
                        2: 50,
                        3: 50
                    }  # steps and thresholds for iterative guidance
                    scale = 0
                else:
                    thresholds = {
                        0: 350,
                        1: 200,
                        2: 200,
                        3: 200
                    }  # steps and thresholds for iterative guidance
                    scale = 300
                step_size_default = 0.2 # default step size for iterative guidance
                step_size_iterative = 0.15
                # If we are in the guidance steps, we will perform the latent optimization
                if (
                    perform_guidance_flag
                    and guidance_begin_step <= i < guidance_end_step
                ):

                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        updated_latents = []
                        for (
                            latent,
                            curr_prompt_embeds,
                            curr_add_text_embeds,
                            curr_add_time_ids,
                        ) in zip(
                            latents,
                            guidance_prompt_embeds,
                            guidance_add_text_embeds,
                            guidance_add_time_ids,
                        ):
                            # Forward pass of denoising with text conditioning
                            latent = latent.unsqueeze(0)
                            curr_prompt_embeds = curr_prompt_embeds.unsqueeze(0)
                            curr_add_text_embeds = curr_add_text_embeds.unsqueeze(0)
                            curr_add_time_ids = curr_add_time_ids.unsqueeze(0)

                            # predict the noise residual
                            added_cond_kwargs = {
                                "text_embeds": curr_add_text_embeds,
                                "time_ids": curr_add_time_ids,
                            }
                            loss = 0
                            if i in thresholds.keys():#################
                                with torch.no_grad():
                                    _ = self.unet(
                                        latent,
                                        t,
                                        encoder_hidden_states=curr_prompt_embeds,
                                        timestep_cond=timestep_cond,
                                        cross_attention_kwargs=self.cross_attention_kwargs,
                                        added_cond_kwargs=added_cond_kwargs,
                                        return_dict=False,
                                    )[0]
                                self.unet.zero_grad()
                                # ---------------- compute cross attention map ----------------
                                cross_attention_maps = (
                                    attention_store.aggregate_attention(
                                        from_where=["up", "mid", "down"],
                                        get_cross=True,
                                    )
                                )
                                cross_attention_maps = cross_attention_maps.permute(
                                    2, 0, 1
                                )
                                # object_attention_map = cross_attention_maps[token_index[-1]]
                                object_attention_maps = []
                                for token in token_index:
                                    attn_map = cross_attention_maps[token]
                                    object_attention_map = self.process_attention_map(attn_map)
                                    object_attention_maps.append(object_attention_map)
                                # object_attention_map = self.merge_tokens_to_eos(object_attention_maps)
                                ca_maps_tensor = torch.stack(object_attention_maps, dim=0)  # [num_words, H, W]
                                object_attention_map = ca_maps_tensor.max(dim=0).values
                                # ------------------------------------------------------------
                                loss, _, _, whole_cluster_masks = total_loss(object_attention_map=object_attention_map,
                                                                             ca_maps_tensor=ca_maps_tensor,
                                                                             whole_cluster_masks=whole_cluster_masks,
                                                                             token_counts=token_counts,
                                                                             scale=scale,
                                                                             iteration=i,
                                                                             temperature=self.softmax_temperature)
                                if output_dir is not None:
                                    # ----------------------
                                    # visialize attention map
                                    # ----------------------
                                    # curr_map = attention_store.all_cross_attention[
                                    #     "down_blocks.2.attentions.1_cross"
                                    # ]
                                    # curr_map = torch.stack(
                                    #     curr_map, dim=0
                                    # )  # shape: [10, 1, 1024, 77]
                                    # curr_map = curr_map.mean(dim=0)  # shape: [1, 1024, 77]
                                    # curr_map_reshape = curr_map.reshape(
                                    #     32, 32, curr_map.shape[-1]
                                    # )
                                    # ca_map_d21 = curr_map_reshape.permute(2, 0, 1)[token_index]
                                    for idx, object_attention_map in enumerate(object_attention_maps):
                                        ca_map_d21 = object_attention_map # 暂时用平均ca
                                        if torch.is_tensor(ca_map_d21):
                                            ca_map = ca_map_d21.detach().cpu().numpy()
                                            plt.figure(figsize=(6, 6))
                                            plt.imshow(ca_map, cmap="viridis")
                                            plt.colorbar()
                                            plt.title("Cross Attention Map")
                                            plt.axis("off")  # 关闭坐标轴
                                            plt.savefig(
                                                f"{output_dir}/{prompt}_step{i}_0_{idx}.png",
                                                bbox_inches="tight",
                                            )
                                            plt.close()
                                    # ----------------------
                                
                                loss, latent = self.perform_iterative_refinement_step(
                                    loss = loss,
                                    latents = latent,
                                    attention_store = attention_store,
                                    whole_cluster_masks = whole_cluster_masks,
                                    curr_prompt_embeds = curr_prompt_embeds,
                                    timestep_cond = timestep_cond,
                                    added_cond_kwargs = added_cond_kwargs,
                                    t = t,
                                    threshold=thresholds[i],
                                    object_token_idx = token_index,
                                    i = i,
                                    step_size = step_size_iterative,
                                    max_refinement_steps=20,
                                    token_counts=token_counts,
                                    scale=scale
                                )
                                # loss = 0
                            else:
                                _ = self.unet(
                                    latent,
                                    t,
                                    encoder_hidden_states=curr_prompt_embeds,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                                self.unet.zero_grad()

                                cross_attention_maps = (
                                    attention_store.aggregate_attention(
                                        from_where=["up", "mid", "down"],
                                        get_cross=True,
                                    )
                                )
                                cross_attention_maps = cross_attention_maps.permute(
                                    2, 0, 1
                                )
                                # object_attention_map = cross_attention_maps[token_index[-1]]
                                object_attention_maps = []
                                for token in token_index:
                                    attn_map = cross_attention_maps[token]
                                    object_attention_map = self.process_attention_map(attn_map)
                                    object_attention_maps.append(object_attention_map)
                                ca_maps_tensor = torch.stack(object_attention_maps, dim=0)  # [num_words, H, W]
                                object_attention_map = ca_maps_tensor.max(dim=0).values
                                loss, _, _, whole_cluster_masks = total_loss(object_attention_map=object_attention_map,
                                                                             ca_maps_tensor=ca_maps_tensor,
                                                                             whole_cluster_masks=whole_cluster_masks,
                                                                             token_counts=token_counts,
                                                                             scale=scale,
                                                                             iteration=i,
                                                                             temperature=self.softmax_temperature)
                                if output_dir is not None:
                                    # ----------------------
                                    for idx, object_attention_map in enumerate(object_attention_maps):
                                        ca_map_d21 = object_attention_map  # 暂时用平均ca
                                        if torch.is_tensor(ca_map_d21):
                                            ca_map = ca_map_d21.detach().cpu().numpy()
                                            plt.figure(figsize=(6, 6))
                                            plt.imshow(ca_map, cmap="viridis")
                                            plt.colorbar()
                                            plt.title("Cross Attention Map")
                                            plt.axis("off")  # 关闭坐标轴
                                            plt.savefig(
                                                f"{output_dir}/{prompt}_step{i}_0_{idx}.png",
                                                bbox_inches="tight",
                                            )
                                            plt.close()
                                    # ----------------------
                               
                            if loss != 0:
                                print(f"Iteration {i} | Loss: {loss:0.4f}")
                                latent = self.update_latent(
                                    latents=latent,
                                    loss=loss,
                                    step_size=step_size_default,  
                                )
                            latent = latent.detach()
                            gc.collect()
                            torch.cuda.empty_cache()

                            updated_latents.append(latent)

                    latents = torch.cat(updated_latents, dim=0)

                # =========================================================
                # Perform Loss Optimization End
                # =========================================================

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                # Unet forward pass
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                # --------------------------------------------
                # visialize attention map BEGIN
                # --------------------------------------------
                if output_dir is not None and i%1 == 0:
                    curr_map = attention_store.all_cross_attention['down_blocks.2.attentions.1_cross']
                    curr_map = torch.stack(curr_map, dim=0)  # shape: [10, 2, 1024, 77]
                    curr_map = curr_map.mean(dim=0)          # shape: [2, 1024, 77]

                    # curr_map = curr_map[1].unsqueeze(0)
                    curr_map = curr_map.mean(dim=0, keepdim=True)  # shape: [1, 1024, 77]
                    curr_map_reshape = curr_map.reshape(
                            32, 32, curr_map.shape[-1]
                    ) # shape: [32, 32, 77]
                    # ca_map_d21 = curr_map_reshape.permute(2, 0, 1)[token_index[-1]] # shape: [32, 32]
                    object_attention_maps = []
                    for token in token_index:
                        attn_map = curr_map_reshape.permute(2, 0, 1)[token]
                        object_attention_map = self.process_attention_map(attn_map)
                        object_attention_maps.append(object_attention_map)
                    ca_maps_tensor = torch.stack(object_attention_maps, dim=0)  # [num_words, H, W]
                    ca_map_d21 = ca_maps_tensor.max(dim=0).values

                    for idx, object_attention_map in enumerate(object_attention_maps):
                        ca_map_d21 = object_attention_map  # 暂时用平均ca
                        if torch.is_tensor(ca_map_d21):
                            ca_map = ca_map_d21.detach().cpu().numpy()
                            plt.figure(figsize=(6, 6))
                            plt.imshow(ca_map, cmap='viridis')
                            plt.colorbar()
                            plt.title("Cross Attention Map")
                            plt.axis('off')  # 关闭坐标轴
                            plt.savefig(f"{output_dir}/{prompt}_step{i}_1_{idx}.png", bbox_inches='tight')
                            plt.close()
                # --------------------------------------------
                # visialize attention map END
                # --------------------------------------------
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
