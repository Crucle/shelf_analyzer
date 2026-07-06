"""
Распознавание бренда товара с помощью мультимодальной модели CLIP
(Contrastive Language-Image Pre-training, OpenAI).

Что значит "мультимодальная" здесь: CLIP обучен сопоставлять изображения
и текстовые описания в одном общем векторном пространстве. Чтобы понять,
какой бренд на фото, не нужен отдельный обучающий датасет с фотографиями
именно этих брендов — можно описать каждый бренд текстовой фразой и/или
показать модели несколько примеров фото этого бренда, и сравнить с этим
фотографию товара: чем ближе картинка к эталону в этом общем
пространстве, тем вероятнее, что это он.

Модель и её веса скачиваются автоматически при первом запуске (нужен
интернет; дальше работает офлайн, веса кешируются на диск). Использована
самая лёгкая версия CLIP — "openai/clip-vit-base-patch32" (~600 МБ вместе
с зависимостями torch).

Про качество распознавания похожих товаров: товары с сильно разным
цветом (Coca-Cola, Sprite, Fanta) CLIP различает уверенно даже по общей
текстовой фразе. А вот визуально похожие товары (например, две
прозрачные бутылки разных марок воды) отличить по одному тексту сложнее.

Два способа это улучшить (можно использовать оба сразу):
  1. Текстовые описания внешнего вида (цвет крышки, этикетки и т.п.) —
     несколько описаний на бренд, см. main.py -> DEFAULT_BRANDS.
  2. Эталонные фото-примеры каждого бренда (few-shot) — если словами
     объяснить разницу сложно, эффективнее просто показать модели
     несколько реальных фото. Это НЕ дообучение весов модели (не нужен
     градиентный спуск и большой датасет) — фото превращаются в векторы
     той же моделью CLIP и используются как ещё один эталон для
     сравнения, наравне с текстом. См. reference_images ниже.

И текстовые описания, и фото-примеры одного бренда усредняются в один
итоговый "эталонный вектор" бренда ("prompt/example ensembling").

Технический нюанс: намеренно используется полный проход через
model(**inputs) -> outputs.image_embeds/text_embeds (как в официальной
документации Hugging Face), а не более короткие методы
get_text_features()/get_image_features(). В части версий библиотеки
transformers эти методы возвращают неожиданный тип объекта — это
известная несовместимость между версиями. Полный проход через
forward() — самый стабильный способ использовать CLIP независимо от
конкретной версии transformers.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from detector import BBox

_MODEL_NAME = "openai/clip-vit-base-patch32"

# Модель и процессор загружаются лениво и только один раз за всё время
# работы программы — сама загрузка занимает несколько секунд, а на каждый
# следующий вызов classify_brands переиспользуется уже готовая модель.
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
    confidence: float  # 0..1, насколько модель уверена в этом бренде


def _crop_to_pil(image: np.ndarray, box: BBox) -> Image.Image:
    crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
    crop_rgb = crop[:, :, ::-1]  # OpenCV хранит BGR, CLIP ожидает RGB
    return Image.fromarray(crop_rgb)


def _to_pil(img) -> Image.Image:
    """Приводит эталонное фото-пример к PIL.Image, если оно ещё не PIL."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        # Предполагаем OpenCV BGR, как и остальной код проекта.
        return Image.fromarray(img[:, :, ::-1]).convert("RGB")
    raise TypeError(f"Не знаю, как превратить {type(img)} в изображение")


def _normalize_brands(brands, prompt_template: str) -> tuple[list[str], list[list[str]]]:
    """
    Приводит `brands` к единому виду: список названий брендов + список
    списков текстовых описаний (по одному подсписку на бренд).

    `brands` может быть:
      - списком строк ["Coca-Cola", "Sprite", ...] — тогда для каждого
        бренда используется один общий prompt_template;
      - словарём {"Coca-Cola": "описание" или ["описание1", "описание2"]}
        — тогда используются переданные описания как есть.
    """
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
    """
    Для каждого найденного товара (box) определяет наиболее вероятный
    бренд из `brands`, используя CLIP.

    brands — список названий (используется общий prompt_template) или
    словарь {название: описание(я)} для более точных текстовых описаний.

    reference_images — необязательный словарь {название_бренда: [фото, ...]}
    с эталонными фото-примерами каждого бренда (few-shot). Фото могут
    быть numpy-массивами (OpenCV BGR) или объектами PIL.Image. Если для
    бренда переданы и текстовые описания, и фото-примеры, они
    усредняются вместе в один эталонный вектор.

    min_confidence — если уверенность модели в лучшем бренде ниже этого
    порога, вернётся `unknown_label` вместо навязанного, но
    низкоуверенного варианта.

    Возвращает список BrandMatch (по одному на каждый box, в том же
    порядке), содержащий имя бренда и уверенность модели (0..1).
    """
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
        # image_embeds/text_embeds — уже нормализованные векторы в общем
        # пространстве CLIP (см. docstring модуля про forward() вместо
        # get_*_features()).
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