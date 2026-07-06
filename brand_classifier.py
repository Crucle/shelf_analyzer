from dataclasses import dataclass

import numpy as np
from PIL import Image

from detector import BBox

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        import torch  # noqa: F401  (проверка, что torch установлен)
        from transformers import CLIPModel, CLIPProcessor

        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _model, _processor


@dataclass
class BrandMatch:
    brand: str
    confidence: float  


def _crop_to_pil(image: np.ndarray, box: BBox) -> Image.Image:
    crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
    crop_rgb = crop[:, :, ::-1]  
    return Image.fromarray(crop_rgb)


def _to_pil(img) -> Image.Image:
    """Приводит эталонное фото-пример к PIL.Image, если оно ещё не PIL."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img[:, :, ::-1]).convert("RGB")
    raise TypeError(f"Не знаю, как превратить {type(img)} в изображение")


def _normalize_brands(brands, prompt_template: str) -> tuple[list[str], list[list[str]]]:
    if isinstance(brands, dict):
        names = list(brands.keys())
        prompt_lists = [
            [v] if isinstance(v, str) else list(v) for v in brands.values()
        ]
    else:
        names = list(brands)
        prompt_lists = [[prompt_template.format(b)] for b in names]
    return names, prompt_lists


def classify_brands(
    image: np.ndarray,
    boxes: list[BBox],
    brands,
    reference_images: dict = None,
    prompt_template: str = "a photo of a bottle of {}",
    min_confidence: float = 0.0,
    unknown_label: str = "не определено",
) -> list[BrandMatch]:
    if not boxes:
        return []

    model, processor = _get_model()
    import torch

    reference_images = reference_images or {}
    brand_names, prompt_lists = _normalize_brands(brands, prompt_template)

    crops = [_crop_to_pil(image, b) for b in boxes]
    num_crops = len(crops)

    ref_images_flat: list[Image.Image] = []
    ref_image_brand_idx: list[int] = []
    for brand_idx, name in enumerate(brand_names):
        for img in reference_images.get(name, []):
            ref_images_flat.append(_to_pil(img))
            ref_image_brand_idx.append(brand_idx)

    all_images = crops + ref_images_flat

    all_prompts: list[str] = []
    prompt_brand_idx: list[int] = []
    for brand_idx, plist in enumerate(prompt_lists):
        for p in plist:
            all_prompts.append(p)
            prompt_brand_idx.append(brand_idx)

    inputs = processor(text=all_prompts, images=all_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.numpy()
        text_embeds = outputs.text_embeds.numpy()
        logit_scale = float(model.logit_scale.exp())

    crop_embeds = image_embeds[:num_crops]
    ref_embeds = image_embeds[num_crops:]

    dim = crop_embeds.shape[1]
    num_brands = len(brand_names)
    brand_proto = np.zeros((num_brands, dim), dtype=np.float64)
    brand_counts = np.zeros(num_brands, dtype=np.int32)

    for emb, idx in zip(text_embeds, prompt_brand_idx):
        brand_proto[idx] += emb
        brand_counts[idx] += 1
    for emb, idx in zip(ref_embeds, ref_image_brand_idx):
        brand_proto[idx] += emb
        brand_counts[idx] += 1

    brand_counts = np.maximum(brand_counts, 1)
    brand_proto = brand_proto / brand_counts[:, None]
    norms = np.linalg.norm(brand_proto, axis=1, keepdims=True)
    brand_proto = brand_proto / np.maximum(norms, 1e-8)

    similarity = (crop_embeds @ brand_proto.T) * logit_scale  # [товары, бренды]

    matches: list[BrandMatch] = []
    for row in similarity:
        exp = np.exp(row - row.max())
        probs = exp / exp.sum()
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        if confidence < min_confidence:
            matches.append(BrandMatch(brand=unknown_label, confidence=confidence))
        else:
            matches.append(BrandMatch(brand=brand_names[best_idx], confidence=confidence))
    return matches
