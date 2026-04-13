import torch
import torch.nn.functional as F
import math
import comfy.samplers
import comfy.utils
import nodes

try:
    import nvvfx
    HAS_NVVFX = True
except ImportError:
    HAS_NVVFX = False

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


# ══════════════════════════════════════════════════════════════
# Helper functions for quality improvements
# ══════════════════════════════════════════════════════════════

def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Hermite interpolation: 3t² - 2t³. Much smoother than linear transitions."""
    return t * t * (3.0 - 2.0 * t)


def create_gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian kernel for convolution-based blur."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel / kernel.sum()


def gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to a [B, C, H, W] tensor using depthwise convolution."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = create_gaussian_kernel_2d(kernel_size, sigma)
    channels = img.shape[1]
    # Depthwise convolution: each channel is blurred independently
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(img.device, img.dtype)
    padding = kernel_size // 2
    return F.conv2d(img, kernel, padding=padding, groups=channels)


def unsharp_mask(image_bhwc: torch.Tensor, strength: float, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply Unsharp Mask sharpening to a [B, H, W, C] image tensor."""
    if strength <= 0:
        return image_bhwc
    # Convert to BCHW for convolution
    img = image_bhwc.permute(0, 3, 1, 2)
    blurred = gaussian_blur(img, kernel_size, sigma)
    # Unsharp mask formula: sharpened = original + strength * (original - blurred)
    sharpened = img + strength * (img - blurred)
    return sharpened.permute(0, 2, 3, 1)


def color_match_tile(refined: torch.Tensor, original: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Match the color statistics (mean/std) of the refined tile to the original.
    This prevents color drift between tiles processed independently by KSampler.
    Operates on [B, H, W, C] tensors.
    """
    if strength <= 0:
        return refined

    orig_mean = original.mean(dim=[1, 2], keepdim=True)
    orig_std = original.std(dim=[1, 2], keepdim=True) + 1e-5
    ref_mean = refined.mean(dim=[1, 2], keepdim=True)
    ref_std = refined.std(dim=[1, 2], keepdim=True) + 1e-5

    # Transfer color statistics from original to refined
    matched = (refined - ref_mean) / ref_std * orig_std + orig_mean
    # Blend between raw refined and color-matched based on strength
    return refined * (1.0 - strength) + matched * strength


def generate_blend_mask_smoothstep(out_tile_h, out_tile_w, eff_overlap, y, actual_h, h, x, actual_w, w):
    """Generate a blend mask using smoothstep (Hermite) interpolation for natural transitions."""
    mask = torch.ones((1, out_tile_h, out_tile_w, 1))

    if eff_overlap > 0:
        if y > 0:
            r = min(eff_overlap, out_tile_h // 2)
            ramp = smoothstep(torch.linspace(0, 1, r))
            mask[:, :r, :, :] *= ramp.view(1, r, 1, 1)
        if y + actual_h < h:
            r = min(eff_overlap, out_tile_h // 2)
            ramp = smoothstep(torch.linspace(1, 0, r))
            mask[:, -r:, :, :] *= ramp.view(1, r, 1, 1)
        if x > 0:
            r = min(eff_overlap, out_tile_w // 2)
            ramp = smoothstep(torch.linspace(0, 1, r))
            mask[:, :, :r, :] *= ramp.view(1, 1, r, 1)
        if x + actual_w < w:
            r = min(eff_overlap, out_tile_w // 2)
            ramp = smoothstep(torch.linspace(1, 0, r))
            mask[:, :, -r:, :] *= ramp.view(1, 1, r, 1)

    return mask


def generate_blend_mask_gaussian(out_tile_h, out_tile_w, eff_overlap, y, actual_h, h, x, actual_w, w):
    """Generate a blend mask using a 2D Gaussian falloff for the smoothest possible transitions."""
    mask = torch.ones((1, out_tile_h, out_tile_w, 1))

    if eff_overlap > 0:
        # Create distance fields from each active edge, then apply Gaussian falloff
        sigma = eff_overlap / 2.5  # Sigma covers ~80% of the overlap region

        if y > 0:
            r = min(eff_overlap, out_tile_h // 2)
            dist = torch.linspace(0, 1, r)
            gauss = 1.0 - torch.exp(-((dist ** 2) / (2 * (0.4 ** 2))))  # Normalized Gaussian CDF-like
            mask[:, :r, :, :] *= gauss.view(1, r, 1, 1)
        if y + actual_h < h:
            r = min(eff_overlap, out_tile_h // 2)
            dist = torch.linspace(1, 0, r)
            gauss = 1.0 - torch.exp(-((dist ** 2) / (2 * (0.4 ** 2))))
            mask[:, -r:, :, :] *= gauss.view(1, r, 1, 1)
        if x > 0:
            r = min(eff_overlap, out_tile_w // 2)
            dist = torch.linspace(0, 1, r)
            gauss = 1.0 - torch.exp(-((dist ** 2) / (2 * (0.4 ** 2))))
            mask[:, :, :r, :] *= gauss.view(1, 1, r, 1)
        if x + actual_w < w:
            r = min(eff_overlap, out_tile_w // 2)
            dist = torch.linspace(1, 0, r)
            gauss = 1.0 - torch.exp(-((dist ** 2) / (2 * (0.4 ** 2))))
            mask[:, :, -r:, :] *= gauss.view(1, 1, r, 1)

    return mask


def generate_blend_mask_linear(out_tile_h, out_tile_w, eff_overlap, y, actual_h, h, x, actual_w, w):
    """Generate a blend mask using simple linear interpolation (legacy behavior)."""
    mask = torch.ones((1, out_tile_h, out_tile_w, 1))

    if eff_overlap > 0:
        if y > 0:
            r = min(eff_overlap, out_tile_h // 2)
            mask[:, :r, :, :] *= torch.linspace(0, 1, r).view(1, r, 1, 1)
        if y + actual_h < h:
            r = min(eff_overlap, out_tile_h // 2)
            mask[:, -r:, :, :] *= torch.linspace(1, 0, r).view(1, r, 1, 1)
        if x > 0:
            r = min(eff_overlap, out_tile_w // 2)
            mask[:, :, :r, :] *= torch.linspace(0, 1, r).view(1, 1, r, 1)
        if x + actual_w < w:
            r = min(eff_overlap, out_tile_w // 2)
            mask[:, :, -r:, :] *= torch.linspace(1, 0, r).view(1, 1, r, 1)

    return mask


# ══════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════

class DLN_GridRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLN_RTX-Grid-Refiner",
            display_name="DLN RTX Tile Refiner",
            description="Splits an image into a grid of smaller tiles, applies KSampler with a lower denoise to each, and merges them back. Best for upscaling or adding high-res details.",
            category="DLN/Image",
            inputs=[
                io.Image.Input("image"),
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("denoise", default=0.25, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("grid_divisions", options=["2x2", "4x4", "8x8", "16x16", "32x32"]),
                io.Int.Input("overlap", default=32, min=0, max=512, step=8, tooltip="Overlap between tiles in pixels to prevent visible seams. A smart minimum is enforced automatically."),
                io.Combo.Input("mode", options=["Refiner (Same Size)", "Upscale (Larger Image)"], default="Refiner (Same Size)", tooltip="Refiner keeps the original resolution but adds details. Upscale increases the final image resolution based on the RTX factor."),
                io.Float.Input("rtx_upscale_factor", default=1.0, min=1.0, max=4.0, step=0.01, tooltip="Upscale factor using Nvidia RTX before refining. 1.0 disables it (requires nvvfx)."),
                io.Combo.Input("rtx_quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH"),
                # ── Quality Enhancement Inputs ──
                io.Combo.Input("blend_mode", options=["Smoothstep", "Gaussian", "Linear"], default="Smoothstep", tooltip="Blending algorithm for tile transitions. Smoothstep and Gaussian are much smoother than Linear."),
                io.Float.Input("sharpening", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Unsharp mask strength applied to the final result. 0=disabled. 0.3-0.5 recommended for crisp details."),
                io.Float.Input("color_match_strength", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Match refined tile colors to the original to prevent color drift between tiles. 0.7-0.9 recommended."),
                io.Float.Input("detail_inject", default=0.0, min=0.0, max=0.5, step=0.01, tooltip="Inject noise into the latent before sampling to encourage micro-detail generation without increasing denoise. 0.05-0.15 recommended."),
                io.Int.Input("context_padding", default=0, min=0, max=128, step=8, tooltip="Extra pixels extracted around each tile for KSampler context. Discarded after processing. 32-64 recommended for better tile coherence."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask to only refine specific areas. Tiles not covered by the mask will be skipped."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The refined and merged image"),
            ],
            hidden=[],
        )

    @classmethod
    def fingerprint_inputs(cls, image, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise, grid_divisions, overlap, mode, rtx_upscale_factor, rtx_quality, blend_mode="Smoothstep", sharpening=0.0, color_match_strength=0.0, detail_inject=0.0, context_padding=0, mask=None):
        return (str(seed), str(steps), str(cfg), sampler_name, scheduler, str(denoise), grid_divisions, str(overlap), mode, str(rtx_upscale_factor), rtx_quality, blend_mode, str(sharpening), str(color_match_strength), str(detail_inject), str(context_padding))

    @classmethod
    def execute(cls, image, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise, grid_divisions, overlap, mode, rtx_upscale_factor, rtx_quality, blend_mode="Smoothstep", sharpening=0.0, color_match_strength=0.0, detail_inject=0.0, context_padding=0, mask=None):
        user_mask = mask
        use_rtx = rtx_upscale_factor > 1.0 and HAS_NVVFX
        
        if rtx_upscale_factor > 1.0 and not HAS_NVVFX:
            print("[DLN RTX Tile Refiner] Warning: RTX Upscaling requested but nvvfx is not installed. Skipping RTX.")
            
        b, h, w, c = image.shape
        grid_n = int(grid_divisions.split("x")[0])
        is_upscale_mode = (mode == "Upscale (Larger Image)")

        # ── Select blend mask generator ──
        blend_generators = {
            "Smoothstep": generate_blend_mask_smoothstep,
            "Gaussian": generate_blend_mask_gaussian,
            "Linear": generate_blend_mask_linear,
        }
        blend_fn = blend_generators.get(blend_mode, generate_blend_mask_smoothstep)
        
        # ── Pre-process mask ──
        if user_mask is not None:
            # user_mask is [B, H_m, W_m] or [H_m, W_m]
            user_mask = user_mask.to(image.device)
            if len(user_mask.shape) == 2:
                user_mask = user_mask.unsqueeze(0) # [1, H_m, W_m]
            if user_mask.shape[1] != h or user_mask.shape[2] != w:
                user_mask = F.interpolate(user_mask.unsqueeze(1), size=(h, w), mode='bilinear').squeeze(1)
            user_mask = user_mask.unsqueeze(-1) # [B, H, W, 1]
        
        # ──────────────────────────────────────────────
        # 1. Determine output dimensions
        # ──────────────────────────────────────────────
        if is_upscale_mode and use_rtx:
            out_h = (int(h * rtx_upscale_factor) // 8) * 8
            out_w = (int(w * rtx_upscale_factor) // 8) * 8
            effective_factor_h = out_h / h
            effective_factor_w = out_w / w
            print(f"[DLN RTX Tile Refiner] Mode: Upscale | Input: {w}x{h} -> Output: {out_w}x{out_h}")
        else:
            out_h, out_w = h, w
            effective_factor_h, effective_factor_w = 1.0, 1.0
            print(f"[DLN RTX Tile Refiner] Mode: Refiner | Resolution: {w}x{h}")

        # ── Create high-res mask for merging ──
        if user_mask is not None:
            if is_upscale_mode:
                final_mask = F.interpolate(
                    user_mask.permute(0, 3, 1, 2),
                    size=(out_h, out_w), mode='bilinear'
                ).permute(0, 2, 3, 1).cpu()
            else:
                final_mask = user_mask.cpu()
        else:
            final_mask = None

        # OPTIMIZATION: out_image and weight_map live on CPU to save VRAM
        # Only the current tile needs GPU memory at any time
        out_image = torch.zeros((b, out_h, out_w, c), dtype=image.dtype)
        weight_map = torch.zeros((b, out_h, out_w, 1), dtype=image.dtype)

        # ──────────────────────────────────────────────
        # 2. Calculate tile grid with smart minimum overlap
        # ──────────────────────────────────────────────
        base_tile_h = h // grid_n
        base_tile_w = w // grid_n

        # IMPROVEMENT 6: Smart minimum overlap — prevent user error with 0 overlap
        min_overlap = max(32, min(base_tile_h, base_tile_w) // 10)
        effective_overlap = max(overlap, min_overlap) if grid_n > 1 else overlap
        if effective_overlap != overlap and grid_n > 1:
            print(f"[DLN RTX Tile Refiner] Smart overlap: {overlap}px -> {effective_overlap}px (minimum for {grid_divisions} grid)")
        
        tile_h = max((base_tile_h + effective_overlap) // 8 * 8, 64)
        tile_w = max((base_tile_w + effective_overlap) // 8 * 8, 64)
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)
        
        step_h = max(tile_h - effective_overlap, 8)
        step_w = max(tile_w - effective_overlap, 8)
        
        y_coords = list(range(0, max(1, h - tile_h + 1), step_h))
        if h > tile_h and y_coords[-1] + tile_h < h:
            y_coords.append(h - tile_h)
            
        x_coords = list(range(0, max(1, w - tile_w + 1), step_w))
        if w > tile_w and x_coords[-1] + tile_w < w:
            x_coords.append(w - tile_w)
            
        total_tiles = len(y_coords) * len(x_coords)
        features = []
        if blend_mode != "Linear":
            features.append(f"Blend:{blend_mode}")
        if sharpening > 0:
            features.append(f"Sharp:{sharpening:.2f}")
        if color_match_strength > 0:
            features.append(f"ColorMatch:{color_match_strength:.2f}")
        if detail_inject > 0:
            features.append(f"DetailInject:{detail_inject:.2f}")
        if context_padding > 0:
            features.append(f"CtxPad:{context_padding}px")
        features_str = f" | Enhancements: {', '.join(features)}" if features else ""
        print(f"[DLN RTX Tile Refiner] Grid: {len(x_coords)}x{len(y_coords)} | Tile Size: {tile_w}x{tile_h} | Overlap: {effective_overlap}px | Total: {total_tiles} tiles{features_str}")

        pbar = comfy.utils.ProgressBar(total_tiles)

        # ──────────────────────────────────────────────
        # 3. OPTIMIZATION: Pre-initialize NVVFX once
        # ──────────────────────────────────────────────
        sr_context = None
        if use_rtx:
            quality_mapping = {
                "LOW": nvvfx.effects.QualityLevel.LOW,
                "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
                "HIGH": nvvfx.effects.QualityLevel.HIGH,
                "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
            }
            selected_quality = quality_mapping.get(rtx_quality, nvvfx.effects.QualityLevel.HIGH)
            sr_context = nvvfx.VideoSuperRes(selected_quality)
            sr_context.__enter__()

        # ──────────────────────────────────────────────
        # 4. Pre-compute blend mask cache
        # ──────────────────────────────────────────────
        cached_mask = None
        cached_mask_size = None

        # IMPROVEMENT 7: Context padding — align to 8px
        ctx_pad = (context_padding // 8) * 8

        try:
            current_tile = 0
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    current_tile += 1
                    
                    actual_h = min(tile_h, h - y)
                    actual_w = min(tile_w, w - x)
                    actual_h = (actual_h // 8) * 8
                    actual_w = (actual_w // 8) * 8
                    
                    if actual_h == 0 or actual_w == 0:
                        print(f"[DLN RTX Tile Refiner] Tile {current_tile}/{total_tiles}: Skipped (too small)")
                        pbar.update(1)
                        continue
                    
                    print(f"[DLN RTX Tile Refiner] Tile {current_tile}/{total_tiles} | Pos: ({x},{y}) | Size: {actual_w}x{actual_h}")
                    
                    # ── Mask check (before extraction to save work) ──
                    if user_mask is not None:
                        tile_mask = user_mask[:, y:y+actual_h, x:x+actual_w, :]
                        if tile_mask.sum() < 0.001:
                            # Skip tile if no mask overlap
                            pbar.update(1)
                            seed += 1
                            continue

                    # ── IMPROVEMENT 7: Extract tile with context padding ──
                    if ctx_pad > 0:
                        # Compute padded extraction region (clamped to image bounds)
                        pad_y1 = max(0, y - ctx_pad)
                        pad_x1 = max(0, x - ctx_pad)
                        pad_y2 = min(h, y + actual_h + ctx_pad)
                        pad_x2 = min(w, x + actual_w + ctx_pad)
                        # Align to 8px
                        pad_ext_h = ((pad_y2 - pad_y1) // 8) * 8
                        pad_ext_w = ((pad_x2 - pad_x1) // 8) * 8
                        pad_y2 = pad_y1 + pad_ext_h
                        pad_x2 = pad_x1 + pad_ext_w

                        if pad_ext_h >= 64 and pad_ext_w >= 64:
                            tile_with_ctx = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :]
                            # Remember the crop offsets to extract the core tile after processing
                            core_y_start = y - pad_y1
                            core_x_start = x - pad_x1
                            core_y_end = core_y_start + actual_h
                            core_x_end = core_x_start + actual_w
                            using_ctx_pad = True
                        else:
                            tile_with_ctx = image[:, y:y+actual_h, x:x+actual_w, :]
                            using_ctx_pad = False
                    else:
                        tile_with_ctx = image[:, y:y+actual_h, x:x+actual_w, :]
                        using_ctx_pad = False

                    # Keep a reference to the original tile for color matching
                    original_tile_ref = image[:, y:y+actual_h, x:x+actual_w, :]
                    
                    # ── RTX Upscale (tile level) ──
                    if use_rtx:
                        enc_h, enc_w = tile_with_ctx.shape[1], tile_with_ctx.shape[2]
                        up_w = max(8, round(enc_w * rtx_upscale_factor / 8) * 8)
                        up_h = max(8, round(enc_h * rtx_upscale_factor / 8) * 8)
                        
                        # Reuse the same NVVFX session, just update dimensions if needed
                        sr_context.output_width = up_w
                        sr_context.output_height = up_h
                        sr_context.load()
                        
                        tile_cuda = tile_with_ctx.cuda().permute(0, 3, 1, 2).contiguous()
                        upscaled_frames = []
                        for frame_idx in range(tile_cuda.shape[0]):
                            dlpack_out = sr_context.run(tile_cuda[frame_idx]).image
                            upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())
                        
                        tile_to_encode = torch.stack(upscaled_frames, dim=0).permute(0, 2, 3, 1).cpu()
                        
                        # Update context crop coords for upscaled space
                        if using_ctx_pad:
                            scale_y = tile_to_encode.shape[1] / enc_h
                            scale_x = tile_to_encode.shape[2] / enc_w
                            core_y_start = int(core_y_start * scale_y)
                            core_x_start = int(core_x_start * scale_x)
                            core_y_end = int(core_y_end * scale_y)
                            core_x_end = int(core_x_end * scale_x)

                        # Free GPU memory immediately
                        del tile_cuda, upscaled_frames
                        torch.cuda.empty_cache()
                        
                        print(f"  -> RTX Upscale: {enc_w}x{enc_h} -> {up_w}x{up_h}")
                    else:
                        tile_to_encode = tile_with_ctx

                    # ── VAE Encode ──
                    tile_latent = vae.encode(tile_to_encode)
                    del tile_to_encode  # Free immediately

                    # ── IMPROVEMENT 5: Noise injection for micro-detail generation ──
                    if detail_inject > 0:
                        noise = torch.randn_like(tile_latent) * detail_inject
                        tile_latent = tile_latent + noise
                        del noise
                    
                    # ── KSampler ──
                    print(f"  -> KSampler (steps:{steps}, denoise:{denoise}, seed:{seed})")
                    sampled_latent_tuple = nodes.common_ksampler(
                        model=model, seed=seed, steps=steps, cfg=cfg,
                        sampler_name=sampler_name, scheduler=scheduler,
                        positive=positive, negative=negative,
                        latent={"samples": tile_latent}, denoise=denoise
                    )
                    sampled_latent = sampled_latent_tuple[0]["samples"]
                    del tile_latent  # Free latent input
                    
                    # ── VAE Decode ──
                    refined_full = vae.decode(sampled_latent)
                    del sampled_latent  # Free latent output

                    # ── IMPROVEMENT 7: Crop out the core tile from the padded result ──
                    if using_ctx_pad:
                        refined_tile = refined_full[:, core_y_start:core_y_end, core_x_start:core_x_end, :]
                        del refined_full
                    else:
                        refined_tile = refined_full

                    # ── Downscale back if Refiner mode ──
                    if use_rtx and not is_upscale_mode:
                        refined_tile = F.interpolate(
                            refined_tile.permute(0, 3, 1, 2),
                            size=(actual_h, actual_w), mode='bicubic', align_corners=False
                        ).permute(0, 2, 3, 1)

                    # ── IMPROVEMENT 4: Color matching to prevent drift between tiles ──
                    if color_match_strength > 0:
                        # Match to original tile (at original or refined resolution)
                        orig_for_match = original_tile_ref.to(refined_tile.device)
                        if orig_for_match.shape[1] != refined_tile.shape[1] or orig_for_match.shape[2] != refined_tile.shape[2]:
                            orig_for_match = F.interpolate(
                                orig_for_match.permute(0, 3, 1, 2),
                                size=(refined_tile.shape[1], refined_tile.shape[2]),
                                mode='bicubic', align_corners=False
                            ).permute(0, 2, 3, 1)
                        refined_tile = color_match_tile(refined_tile, orig_for_match, color_match_strength)
                        del orig_for_match

                    # ── Coordinate mapping ──
                    out_y = int(y * effective_factor_h)
                    out_x = int(x * effective_factor_w)
                    out_tile_h = refined_tile.shape[1]
                    out_tile_w = refined_tile.shape[2]

                    # ── Generate or reuse blend mask ──
                    mask_key = (out_tile_h, out_tile_w, y > 0, y + actual_h < h, x > 0, x + actual_w < w, blend_mode)
                    
                    if cached_mask_size != mask_key:
                        eff_ovlp = int(effective_overlap * effective_factor_h)
                        blend_mask = blend_fn(out_tile_h, out_tile_w, eff_ovlp, y, actual_h, h, x, actual_w, w)
                        cached_mask = blend_mask
                        cached_mask_size = mask_key
                    else:
                        blend_mask = cached_mask

                    # ── Safe fit bounds ──
                    fit_h = min(out_tile_h, out_image.shape[1] - out_y)
                    fit_w = min(out_tile_w, out_image.shape[2] - out_x)
                    
                    if fit_h <= 0 or fit_w <= 0:
                        print(f"  -> Skipped merge: tile extends outside output bounds")
                        pbar.update(1)
                        seed += 1
                        continue

                    # ── Merge on CPU (saves VRAM) ──
                    refined_cpu = refined_tile[:, :fit_h, :fit_w, :].cpu()
                    mask_cpu = blend_mask[:, :fit_h, :fit_w, :].clone()
                    
                    if final_mask is not None:
                        # Extract the user mask part corresponding to the output tile
                        user_mask_tile = final_mask[:, out_y:out_y+fit_h, out_x:out_x+fit_w, :]
                        mask_cpu *= user_mask_tile
                    
                    out_image[:, out_y:out_y+fit_h, out_x:out_x+fit_w, :] += refined_cpu * mask_cpu
                    weight_map[:, out_y:out_y+fit_h, out_x:out_x+fit_w, :] += mask_cpu
                    
                    # Free GPU tile memory
                    del refined_tile, refined_cpu
                    torch.cuda.empty_cache()
                    
                    pbar.update(1)
                    seed += 1
                    
        finally:
            # OPTIMIZATION: Always close the NVVFX context
            if sr_context is not None:
                sr_context.__exit__(None, None, None)
                print("[DLN RTX Tile Refiner] NVVFX session closed.")

        # ──────────────────────────────────────────────
        # 5. Normalize and finalize (all on CPU)
        # ──────────────────────────────────────────────
        print("[DLN RTX Tile Refiner] Merging all tiles...")
        
        weight_map_safe = torch.clamp(weight_map, min=1e-5)
        out_image = out_image / weight_map_safe
        del weight_map, weight_map_safe  # Free
        
        # Fallback for uncovered areas
        if is_upscale_mode and use_rtx:
            bg = F.interpolate(
                image.cpu().permute(0, 3, 1, 2),
                size=(out_h, out_w), mode='bicubic', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            bg = image.cpu()
            
        if user_mask is not None:
            # Final blend Refined vs Background using High-res User Mask
            out_image = out_image * final_mask + bg * (1.0 - final_mask)
        else:
            out_image = torch.where(out_image.abs().sum(dim=-1, keepdim=True) > 0.001, out_image, bg)

        # ── IMPROVEMENT 3: Sharpening post-process (Unsharp Mask) ──
        if sharpening > 0:
            print(f"[DLN RTX Tile Refiner] Applying Unsharp Mask (strength: {sharpening:.2f})...")
            out_image = unsharp_mask(out_image, strength=sharpening, kernel_size=5, sigma=1.0)
        
        out_image = torch.clamp(out_image, 0.0, 1.0)
        
        print(f"[DLN RTX Tile Refiner] Done! Output: {out_image.shape[2]}x{out_image.shape[1]}")
        return io.NodeOutput(out_image)

class DLNRefinerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DLN_GridRefiner]

async def comfy_entrypoint() -> DLNRefinerExtension:
    return DLNRefinerExtension()
