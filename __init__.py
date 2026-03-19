import torch
import torch.nn.functional as F
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
                io.Int.Input("overlap", default=32, min=0, max=512, step=8, tooltip="Overlap between tiles in pixels to prevent visible seams."),
                io.Combo.Input("mode", options=["Refiner (Same Size)", "Upscale (Larger Image)"], default="Refiner (Same Size)", tooltip="Refiner keeps the original resolution but adds details. Upscale increases the final image resolution based on the RTX factor."),
                io.Float.Input("rtx_upscale_factor", default=1.0, min=1.0, max=4.0, step=0.01, tooltip="Upscale factor using Nvidia RTX before refining. 1.0 disables it (requires nvvfx)."),
                io.Combo.Input("rtx_quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH"),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask to only refine specific areas. Tiles not covered by the mask will be skipped."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The refined and merged image"),
            ],
            hidden=[],
        )

    @classmethod
    def fingerprint_inputs(cls, image, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise, grid_divisions, overlap, mode, rtx_upscale_factor, rtx_quality, mask=None):
        return (str(seed), str(steps), str(cfg), sampler_name, scheduler, str(denoise), grid_divisions, str(overlap), mode, str(rtx_upscale_factor), rtx_quality)

    @classmethod
    def execute(cls, image, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise, grid_divisions, overlap, mode, rtx_upscale_factor, rtx_quality, mask=None):
        user_mask = mask
        use_rtx = rtx_upscale_factor > 1.0 and HAS_NVVFX
        
        if rtx_upscale_factor > 1.0 and not HAS_NVVFX:
            print("[DLN RTX Tile Refiner] Warning: RTX Upscaling requested but nvvfx is not installed. Skipping RTX.")
            
        b, h, w, c = image.shape
        grid_n = int(grid_divisions.split("x")[0])
        is_upscale_mode = (mode == "Upscale (Larger Image)")
        
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
        # 2. Calculate tile grid
        # ──────────────────────────────────────────────
        base_tile_h = h // grid_n
        base_tile_w = w // grid_n
        
        tile_h = max((base_tile_h + overlap) // 8 * 8, 64)
        tile_w = max((base_tile_w + overlap) // 8 * 8, 64)
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)
        
        step_h = max(tile_h - overlap, 8)
        step_w = max(tile_w - overlap, 8)
        
        y_coords = list(range(0, max(1, h - tile_h + 1), step_h))
        if h > tile_h and y_coords[-1] + tile_h < h:
            y_coords.append(h - tile_h)
            
        x_coords = list(range(0, max(1, w - tile_w + 1), step_w))
        if w > tile_w and x_coords[-1] + tile_w < w:
            x_coords.append(w - tile_w)
            
        total_tiles = len(y_coords) * len(x_coords)
        print(f"[DLN RTX Tile Refiner] Grid: {len(x_coords)}x{len(y_coords)} | Tile Size: {tile_w}x{tile_h} | Overlap: {overlap}px | Total: {total_tiles} tiles")

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
        # 4. OPTIMIZATION: Pre-compute blend mask
        #    (reuse for all tiles of the same size)
        # ──────────────────────────────────────────────
        cached_mask = None
        cached_mask_size = None

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
                    
                    # ── Extract tile ──
                    tile = image[:, y:y+actual_h, x:x+actual_w, :]
                    
                    # ── Mask check ──
                    if user_mask is not None:
                        tile_mask = user_mask[:, y:y+actual_h, x:x+actual_w, :]
                        if tile_mask.sum() < 0.001:
                            # Skip tile if no mask overlap
                            pbar.update(1)
                            seed += 1
                            continue
                    
                    # ── RTX Upscale (tile level) ──
                    if use_rtx:
                        up_w = max(8, round(actual_w * rtx_upscale_factor / 8) * 8)
                        up_h = max(8, round(actual_h * rtx_upscale_factor / 8) * 8)
                        
                        # Reuse the same NVVFX session, just update dimensions if needed
                        sr_context.output_width = up_w
                        sr_context.output_height = up_h
                        sr_context.load()
                        
                        tile_cuda = tile.cuda().permute(0, 3, 1, 2).contiguous()
                        upscaled_frames = []
                        for frame_idx in range(tile_cuda.shape[0]):
                            dlpack_out = sr_context.run(tile_cuda[frame_idx]).image
                            upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())
                        
                        tile_to_encode = torch.stack(upscaled_frames, dim=0).permute(0, 2, 3, 1).cpu()
                        
                        # Free GPU memory immediately
                        del tile_cuda, upscaled_frames
                        torch.cuda.empty_cache()
                        
                        print(f"  -> RTX Upscale: {actual_w}x{actual_h} -> {up_w}x{up_h}")
                    else:
                        tile_to_encode = tile

                    # ── VAE Encode ──
                    tile_latent = vae.encode(tile_to_encode)
                    del tile_to_encode  # Free immediately
                    
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
                    refined_tile = vae.decode(sampled_latent)
                    del sampled_latent  # Free latent output
                    
                    # ── Downscale back if Refiner mode ──
                    if use_rtx and not is_upscale_mode:
                        refined_tile = F.interpolate(
                            refined_tile.permute(0, 3, 1, 2),
                            size=(actual_h, actual_w), mode='bicubic', align_corners=False
                        ).permute(0, 2, 3, 1)

                    # ── Coordinate mapping ──
                    out_y = int(y * effective_factor_h)
                    out_x = int(x * effective_factor_w)
                    out_tile_h = refined_tile.shape[1]
                    out_tile_w = refined_tile.shape[2]

                    # ── Generate or reuse blend mask ──
                    mask_key = (out_tile_h, out_tile_w, y > 0, y + actual_h < h, x > 0, x + actual_w < w)
                    
                    if cached_mask_size != mask_key:
                        mask = torch.ones((1, out_tile_h, out_tile_w, 1))
                        
                        if overlap > 0:
                            eff_overlap = int(overlap * effective_factor_h)
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
                        
                        cached_mask = mask
                        cached_mask_size = mask_key
                    else:
                        mask = cached_mask

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
                    mask_cpu = mask[:, :fit_h, :fit_w, :].clone()
                    
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
        
        out_image = torch.clamp(out_image, 0.0, 1.0)
        
        print(f"[DLN RTX Tile Refiner] Done! Output: {out_image.shape[2]}x{out_image.shape[1]}")
        return io.NodeOutput(out_image)

class DLNRefinerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DLN_GridRefiner]

async def comfy_entrypoint() -> DLNRefinerExtension:
    return DLNRefinerExtension()
