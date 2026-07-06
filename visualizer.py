"""Отрисовка результатов: рамки товаров (цвет = бренд, подписан текстом),
нарушения дополнительно обводятся красным."""
import cv2
import numpy as np

from detector import BBox

_PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (238, 130, 238),
    (255, 215, 0),
    (0, 206, 209),
    (255, 140, 0),
    (147, 112, 219),
]


def _color_for_label(label: str) -> tuple:
    """
    Детерминированно назначает цвет по строковой метке (названию бренда):
    один и тот же бренд всегда получает один и тот же цвет в пределах
    одного запуска программы. Обычный список/словарь тут не подходит,
    т.к. цвет должен не зависеть от порядка появления брендов на фото.
    """
    idx = hash(label) % len(_PALETTE)
    return _PALETTE[idx]


def draw_result(
    image: np.ndarray,
    boxes: list[BBox],
    labels: list[str],
    violation_labels: set[str],
) -> np.ndarray:
    out = image.copy()
    for box, label in zip(boxes, labels):
        color = _color_for_label(label)
        cv2.rectangle(out, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
        # Если рамка — это группа из нескольких вплотную стоящих товаров
        # (см. BBox.count в detector.py), показываем оценку количества
        # прямо на подписи, чтобы не выдавать её молча за один товар.
        text = f"{label} ×{box.count}" if box.count > 1 else str(label)
        cv2.putText(
            out,
            text,
            (box.x, max(box.y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        if label in violation_labels:
            cv2.rectangle(
                out,
                (box.x - 3, box.y - 3),
                (box.x + box.w + 3, box.y + box.h + 3),
                (0, 0, 255),
                2,
            )
    return out