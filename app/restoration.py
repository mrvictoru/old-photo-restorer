import io
import os
import uuid
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import DiffusionPipeline, FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

def _average_saturation(img: Image.Image) -> float:
    """Rudimentary check for color saturation; returns 0 for gray images."""
    if img.mode != "RGB":
        im = img.convert("RGB")
    else:
        im = img
    hsv = np.array(im.convert("HSV"))
    sat = hsv[..., 1].astype(np.float32) / 255.0
    return float(np.mean(sat))

def _looks_grayscale(img: Image.Image, sat_threshold: float = 0.06) -> bool:
    """Detect grayscale or low-saturation images."""
    if img.mode in ("1", "L", "LA", "I", "I;16", "P"):
        return True
    sat = _average_saturation(img)
    return sat < sat_threshold

def _max_size_resize(img: Image.Image, max_side: int = 1024) -> Tuple[Image.Image, float]:
    """Resize so the longest side is <= max_side, preserving aspect ratio. Returns (image, scale_factor)."""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img, 1.0
    scale = max_side / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS), scale

def _simple_upscale(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Simple high-quality upscaling via Lanczos. Replace with a learned upscaler if desired."""
    if factor <= 1.0:
        return img
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)

class KontextRestorer:
    def __init__(self,
        model_id: str = "black-forest-labs/FLUX.1-Kontext-dev",
        access_token: str = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):  
        self.model_id = model_id
        self.access_token = access_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is None:
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Load model
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            token=self.access_token
        )
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # lazy-loaded upscaler state (Flux controlnet + pipeline)
        self._upscaler_controlnet = None
        self._upscaler_pipe = None
        self._upscaler_model_ids = {
            "controlnet": "jasperai/Flux.1-dev-Controlnet-Upscaler",
            "base": "black-forest-labs/FLUX.1-dev",
        }

    def build_prompt(self, user_prompt: str, needs_colorization: bool) -> Tuple[str, str]:
        base_prompt = (
            "Restore an old photograph: repair scratches, reduce noise, remove compression artifacts, "
            "recover fine details and textures. Preserve identity, facial features, pose, composition, "
            "and lighting. Maintain realism and period-accurate appearance. No stylization."
        )
        color_prompt = (
            " Colorize realistically with natural skin tones, hair, textiles, foliage, sky, and materials. "
            "Avoid over-saturation. Respect original luminance and contrast. When ambiguous, choose plausible colors."
        ) if needs_colorization else ""

        enhancement_prompt = (
            " Enhance clarity slightly while retaining authentic grain. Avoid plastic skin or over-sharpening."
        )

        full_prompt = " ".join(
            p for p in [base_prompt, color_prompt, enhancement_prompt, user_prompt or ""] if p
        )

        negative_prompt = (
            "cartoonish, over-saturated, plastic skin, waxy, over-sharpened halos, color bleed, "
            "posterization, extra fingers, warped faces, distorted anatomy, heavy stylization, low-res, blurry"
        )
        return full_prompt, negative_prompt

    def _call_with_image_kw(self, image: Image.Image, **kwargs):
        """
        Try common parameter names for context/image input since pipelines may differ.
        """
        # First try standard img2img style
        try:
            return self.pipe(image=image, **kwargs)
        except TypeError:
            pass
        # Try 'context' (Kontext)
        try:
            return self.pipe(context=image, **kwargs)
        except TypeError:
            pass
        # Try 'context_image'
        try:
            return self.pipe(context_image=image, **kwargs)
        except TypeError:
            pass
        # As a last resort, try positional call (not ideal)
        try:
            return self.pipe(image, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to call the FLUX.1-Kontext pipeline with the input image. "
                "The parameter name for the context image may differ in your diffusers version."
            ) from e

    def restore(
        self,
        img: Image.Image,
        user_prompt: str = "",
        guidance_scale: float = 3.5,
        strength: float = 0.35,
        num_inference_steps: int = 28,
        upscale_factor: float = 2.0,
        max_input_side: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        # Convert to RGB and resize to manageable compute size
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        resized, _ = _max_size_resize(rgb, max_input_side)

        needs_colorization = _looks_grayscale(resized)
        prompt, negative_prompt = self.build_prompt(user_prompt, needs_colorization)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Call the pipeline
        result = self._call_with_image_kw(
            resized,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        out_img: Image.Image = result.images[0]

        # Upscale using Flux upscaler with fallback to simple Lanczos upscaling
        try:
            up_img = self._flux_upscale(
                out_img,
                factor=upscale_factor,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=0.6,
            )
            # If flux returned something unexpected, fall back.
            if not isinstance(up_img, Image.Image):
                raise RuntimeError("Flux upscaler returned non-image result")
            out_img = up_img
        except Exception:
            # On any error, fall back to the simple Lanczos upscaler
            out_img = _simple_upscale(out_img, upscale_factor)
 
        return out_img

    def _load_upscaler(self):
        """Lazy-load the Flux controlnet upscaler pipeline. May raise on load failure."""
        if self._upscaler_pipe is not None:
            return

        # Choose dtype for upscaler: prefer configured torch_dtype but force float32 on CPU.
        dtype = self.torch_dtype
        if self.device == "cpu":
            dtype = torch.float32

        # Load controlnet and Flux pipeline (may be large; done lazily)
        self._upscaler_controlnet = FluxControlNetModel.from_pretrained(
            self._upscaler_model_ids["controlnet"],
            torch_dtype=dtype,
            token=self.access_token,
        )
        self._upscaler_pipe = FluxControlNetPipeline.from_pretrained(
            self._upscaler_model_ids["base"],
            controlnet=self._upscaler_controlnet,
            torch_dtype=dtype,
            token=self.access_token,
        )
        self._upscaler_pipe.to(self.device)
        self._upscaler_pipe.set_progress_bar_config(disable=True)

    def _flux_upscale(
        self,
        image: Image.Image,
        factor: float = 2.0,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        controlnet_conditioning_scale: float = 0.6,
    ) -> Image.Image:
        """Upscale using the Flux ControlNet upscaler. Falls back to Lanczos _simple_upscale on error."""
        if factor <= 1.0:
            return image

        try:
            self._load_upscaler()
        except Exception:
            # If loading fails, fallback to simple upscaler
            return _simple_upscale(image, factor)

        # compute target size and prepare control image (Flux expects the upscaled control image)
        w, h = image.size
        new_w, new_h = int(round(w * factor)), int(round(h * factor))
        control_image = image.resize((new_w, new_h), Image.LANCZOS)

        try:
            result = self._upscaler_pipe(
                prompt="",
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=new_h,
                width=new_w,
            )
            return result.images[0]
        except Exception:
            # If inference fails for any reason, fall back to simple resize.
            return _simple_upscale(image, factor)
