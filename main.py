import argparse
import json

import cv2
import numpy as np

from brand_classifier import classify_brands
from detector import BBox, auto_crop, detect_products
from layout_checker import check_layout, group_into_rows
from planogram import compare_to_planogram, load_planogram, summarize as summarize_planogram
from report import build_layout_report
from visualizer import draw_result

DEFAULT_BRANDS = {
    "Coca-Cola": [
        "a photo of a dark cola soft drink bottle with a red logo and red bottle cap",
        "a photo of a Coca-Cola soda bottle",
    ],
    "Sprite": [
        "a photo of a clear bottle of bright green lemon-lime soda with a green bottle cap",
        "a photo of a Sprite soda bottle",
    ],
    "Fanta": [
        "a photo of an orange soda bottle with an orange label and orange bottle cap",
        "a photo of a Fanta soda bottle",
    ],
    "Schweppes": [
        "a photo of a pale golden tonic water bottle with a dark rectangular label and yellow bottle cap",
        "a photo of a Schweppes bottle",
    ],
    "BonAqua": [
        "a photo of a clear still mineral water bottle with a blue label and blue bottle cap",
        "a photo of a BonAqua water bottle",
    ],
}


def _imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except FileNotFoundError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Аналогично _imread_unicode, но для сохранения файла."""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True


def load_reference_images(folder: str, brand_names: list[str] = None) -> dict:
    import os

    result: dict = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Папка с эталонными фото не найдена: {folder}")

    for entry in sorted(os.listdir(folder)):
        brand_dir = os.path.join(folder, entry)
        if not os.path.isdir(brand_dir):
            continue
        if brand_names is not None and entry not in brand_names:
            continue

        images = []
        for fname in sorted(os.listdir(brand_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = _imread_unicode(os.path.join(brand_dir, fname))
            if img is not None:
                images.append(img)
        if images:
            result[entry] = images

    return result


def analyze_image(
    image: np.ndarray,
    brands: list[str] = None,
    rows: int = 1,
    min_confidence: float = 0.0,
    reference_images: dict = None,
):
   
    brands = brands or DEFAULT_BRANDS

    boxes = _detect_all_rows(image, rows)
    if not boxes:
        return image, [], [], [], []

    matches = classify_brands(
        image, boxes, brands, reference_images=reference_images, min_confidence=min_confidence
    )
    labels = [m.brand for m in matches]
    violations = check_layout(boxes, labels)

    return image, boxes, labels, violations, matches


def analyze(
    image_path: str,
    brands: list[str] = None,
    crop=None,
    rows: int = 1,
    reference_images: dict = None,
    use_auto_crop: bool = True,
):
    image = _imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {image_path}")

    if crop is not None:
        x1, y1, x2, y2 = crop
        image = image[y1:y2, x1:x2]
    elif use_auto_crop:
        x1, y1, x2, y2 = auto_crop(image)
        image = image[y1:y2, x1:x2]

    return analyze_image(image, brands, rows, reference_images=reference_images)


def _detect_all_rows(image, rows: int) -> list[BBox]:
    
    h_img = image.shape[0]
    row_height = h_img // rows

    all_boxes: list[BBox] = []
    for i in range(rows):
        y1 = i * row_height
        y2 = h_img if i == rows - 1 else (i + 1) * row_height
        band = image[y1:y2]
        band_boxes = detect_products(band)
        for b in band_boxes:
            all_boxes.append(BBox(b.x, b.y + y1, b.w, b.h, count=b.count))
    return all_boxes


def labels_by_row(boxes: list[BBox], labels: list[str]) -> list:
    rows_idx = group_into_rows(boxes)
    result = []
    for row in rows_idx:
        row_labels = []
        for idx in row:
            row_labels.extend([labels[idx]] * max(boxes[idx].count, 1))
        result.append(row_labels)
    return result


def main():
    parser = argparse.ArgumentParser(description="Анализатор корректности выкладки товаров")
    parser.add_argument("image", help="Путь к фото прилавка")
    parser.add_argument("--out", default="result.jpg", help="Куда сохранить размеченное фото")
    parser.add_argument(
        "--brands",
        default=None,
        help=(
            "Список брендов через запятую, которые нужно искать на фото, "
            'например --brands "Coca-Cola,Sprite,Fanta". Если не указать — '
            "используется список по умолчанию (напитки из тестового фото)."
        ),
    )
    parser.add_argument(
        "--crop",
        default=None,
        help=(
            "Обрезать фото перед анализом до области x1,y1,x2,y2 "
            "(в пикселях). Полезно, если в кадр попал соседний стеллаж "
            "или лишний фон, мешающий детекции — например: "
            "--crop 440,0,1040,768. Если не указать, программа попробует "
            "обрезать фон автоматически (см. --no-auto-crop)."
        ),
    )
    parser.add_argument(
        "--no-auto-crop",
        action="store_true",
        help=(
            "Отключить автоматическую обрезку фона (по умолчанию, если "
            "--crop не указан вручную, программа сама пытается найти "
            "область прилавка по резкости изображения — см. README, "
            "раздел про автообрезку). Полезно отключить, если "
            "автоматическая обрезка ошибается на вашем конкретном фото."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help=(
            "Сколько физических полок на фото (по высоте). По умолчанию 1 "
            "(фото одной полки). Если на фото весь стеллаж — укажите "
            "фактическое число полок, например --rows 3: кадр будет "
            "поделён на столько же равных горизонтальных полос, и на "
            "каждой детекция запустится отдельно. Это единственный "
            "параметр, который пока нельзя определить автоматически "
            "надёжно (см. README) — приходится указывать вручную."
        ),
    )
    parser.add_argument(
        "--references",
        default=None,
        help=(
            "Папка с эталонными фото-примерами брендов для более точного "
            "распознавания похожих товаров (few-shot). Структура — по "
            "подпапке на бренд, см. main.py -> load_reference_images(). "
            "Например: --references ./examples"
        ),
    )
    parser.add_argument(
        "--planogram",
        default=None,
        help=(
            "Путь к JSON-файлу с эталонной планограммой — список списков "
            "брендов, один список на полку сверху вниз, например: "
            '[["Coca-Cola","Coca-Cola","Sprite"],["Fanta","BonAqua"]]. '
            "Если указан, программа дополнительно сравнит фактическую "
            "выкладку с планограммой и найдёт отсутствующие, лишние "
            "товары и нарушения порядка (см. planogram.py)."
        ),
    )
    args = parser.parse_args()

    crop = None
    if args.crop:
        x1, y1, x2, y2 = (int(v) for v in args.crop.split(","))
        crop = (x1, y1, x2, y2)

    brands = [b.strip() for b in args.brands.split(",")] if args.brands else None

    reference_images = None
    if args.references:
        brand_names = list(brands) if brands else list(DEFAULT_BRANDS.keys())
        reference_images = load_reference_images(args.references, brand_names)
        if reference_images:
            print(
                "Загружены эталонные фото: "
                + ", ".join(f"{k} ({len(v)} шт.)" for k, v in reference_images.items())
            )
        else:
            print(f"⚠ В папке {args.references} не найдено подходящих эталонных фото.")

    if crop is None and not args.no_auto_crop:
        print("Область обрезки не указана — пробую найти прилавок автоматически...")

    image, boxes, labels, violations, matches = analyze(
        args.image, brands, crop, args.rows, reference_images, use_auto_crop=not args.no_auto_crop
    )
    image_area = image.shape[0] * image.shape[1]
    report = build_layout_report(image_area, boxes, labels, violations)

    print("=" * 60)
    print("ОТЧЁТ О ВЫКЛАДКЕ")
    print("=" * 60)
    print(f"Соответствие выкладке:   {report['compliance_percent']}%")
    print(f"Критических нарушений:   {report['critical_violations_count']}")
    print(f"Полок распознано:        {report['shelves_recognized']}")
    print(f"Товаров найдено:         {report['total_products']}" + (" (оценочно)" if report["count_is_estimated"] else ""))
    print(f"Позиций (рамок) найдено: {report['positions_detected']}")
    print()
    print(report["summary"])

    if report["warnings"]:
        print()
        print("ПРЕДУПРЕЖДЕНИЯ:")
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report["critical_violations"]:
        print()
        print("КРИТИЧЕСКИЕ НАРУШЕНИЯ:")
        for v in report["critical_violations"]:
            print(f"  — {v}")

    if args.planogram:
        planogram = load_planogram(args.planogram)
        actual_rows = labels_by_row(boxes, labels)
        diffs = compare_to_planogram(actual_rows, planogram)
        plano_summary = summarize_planogram(diffs)

        print()
        print("=" * 60)
        print("СРАВНЕНИЕ С ПЛАНОГРАММОЙ")
        print("=" * 60)
        print(f"Соответствие планограмме: {plano_summary['match_percent']}%")
        print(f"Отсутствует товаров:      {plano_summary['total_missing']}")
        print(f"Лишних товаров:           {plano_summary['total_extra']}")
        print(f"Полок с неверным порядком: {plano_summary['shelves_with_order_issues']}")
        for d in diffs:
            print(f"\nПолка {d.shelf_number}:")
            print(f"  Ожидалось: {d.expected}")
            print(f"  Найдено:   {d.actual}")
            if d.missing:
                print(f"  ⚠ Отсутствуют: {d.missing}")
            if d.extra:
                print(f"  ⚠ Лишние: {d.extra}")
            if not d.order_correct and not d.missing and not d.extra:
                print("  ⚠ Неправильный порядок расположения")
            if not d.missing and not d.extra and d.order_correct:
                print("  ✓ Соответствует планограмме")

        report["planogram_comparison"] = {
            **plano_summary,
            "shelves": [
                {
                    "shelf_number": d.shelf_number,
                    "expected": d.expected,
                    "actual": d.actual,
                    "missing": d.missing,
                    "extra": d.extra,
                    "order_correct": d.order_correct,
                }
                for d in diffs
            ],
        }

    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if boxes:
        violation_labels = {v.label for v in violations}
        result_img = draw_result(image, boxes, labels, violation_labels)
        _imwrite_unicode(args.out, result_img)
        print(f"\nРазмеченное фото сохранено: {args.out}")


if __name__ == "__main__":
    main()
