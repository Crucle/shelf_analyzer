"""
Модуль детекции отдельных товаров на фото прилавка.

Готового датасета с размеченными товарами пока нет, поэтому используется
классический подход на OpenCV без обучения.

Алгоритм — анализ проекции границ (edge projection), а не поиск замкнутых
контуров. Причина: у товаров с "плотной" этикеткой (много текста/графики,
как у колы) или у прозрачных бутылок (вода) не получается выделить один
целый замкнутый контур — слишком много внутренних границ или слишком мало
внешних. Зато почти всегда можно увидеть, в каких X-диапазонах на полке
вообще есть хоть какие-то границы (это и есть товар), а где пусто (это
фон/промежуток между товарами) — такая проекция гораздо устойчивее.

Работает надёжно на фото ОДНОЙ полки. Для фото с несколькими полками
подряд нужно предварительно разрезать на полосы (см. main.py --crop) —
автоматически найти границы между полками по самому фото пока не удалось
сделать надёжно (кромка полки часто прерывается ценниками), это
зафиксировано как известное ограничение и направление для развития.

Отдельный случай — товары стоят вплотную друг к другу без видимого
промежутка (обычная ситуация на плотно заполненной полке в реальном
магазине). Тогда проекция границ не находит разрыва внутри одного
бренда, но между РАЗНЫМИ брендами почти всегда виден резкий переход
цвета — на нём и основано доразбиение (см. _split_by_color). Внутри
получившегося цветового блока количество отдельных товаров не
определяется поштучно, а лишь оценивается по ширине блока (см.
_estimate_count) — это принципиально приблизительная оценка, а не
точный подсчёт, и это специально отражено в отчёте (см. main.py/report.py).

Когда появится размеченный датасет — эту функцию можно заменить на вызов
обученной модели детекции (например YOLOv8). Сигнатура
(image -> list[BBox]) при этом останется прежней, и весь остальной
пайплайн (features -> clustering -> layout_checker) менять не придётся.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Типичное отношение ширины одного товара к его высоте — используется
# только для оценки количества товаров внутри широкого цветового блока,
# когда бутылки стоят вплотную и разделить их поштучно не получается.
# Подобрано по фактическим измерениям одиночных бутылок на тестовых фото
# (реальный разброс примерно 0.2-0.5, это грубое приближение).
_TYPICAL_ITEM_ASPECT = 0.3


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    # Оценка количества отдельных единиц товара внутри этой рамки. Для
    # обычного случая (товар отделён от соседей видимым промежутком) это
    # всегда 1 — рамка и есть один товар. Больше 1 — только когда рамка
    # получена как широкий цветовой блок из плотно стоящих одинаковых
    # товаров без зазора (см. _estimate_count), и это ПРИБЛИЗИТЕЛЬНАЯ
    # оценка по ширине блока, а не точный поштучный подсчёт.
    count: int = 1

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self):
        return self.w * self.h


def _remove_long_horizontal_lines(edges: np.ndarray, min_length: int) -> np.ndarray:
    """
    Убирает из карты границ длинные горизонтальные линии — кромки полки и
    верхние границы ценников, а не контуры товаров. Иначе такая линия
    "засоряет" всю проекцию по столбцам одинаково высоким значением, и
    промежутки между товарами перестают быть видны.
    """
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.bitwise_and(edges, cv2.bitwise_not(horiz_lines))


def _find_occupied_ranges(
    profile: np.ndarray,
    thresh_frac: float,
    min_len_frac: float,
    gap_merge: int,
) -> list[tuple[int, int]]:
    """
    Находит непрерывные диапазоны индексов, где сигнал (сумма границ по
    столбцу или по строке) заметно выше фона. Используется дважды: сначала
    по столбцам (находим сами товары на полке), потом по строкам внутри
    каждого найденного столбца (находим вертикальную протяжённость товара).
    """
    total_len = len(profile)
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    thresh = max(5.0, thresh_frac * smooth.max())
    occupied = smooth > thresh

    raw_ranges: list[list[int]] = []
    start = None
    for i, is_occupied in enumerate(occupied):
        if is_occupied and start is None:
            start = i
        elif not is_occupied and start is not None:
            raw_ranges.append([start, i])
            start = None
    if start is not None:
        raw_ranges.append([start, total_len])

    # Соседние диапазоны с маленьким разрывом между ними — это скорее всего
    # один и тот же товар, у которого в середине случайно мало границ
    # (гладкий однотонный участок упаковки), а не два разных товара.
    merged: list[list[int]] = []
    for r in raw_ranges:
        if merged and r[0] - merged[-1][1] <= gap_merge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    min_len = min_len_frac * total_len
    return [(r[0], r[1]) for r in merged if (r[1] - r[0]) >= min_len]


def _split_by_color(
    image: np.ndarray,
    x1: int,
    x2: int,
    diff_threshold: float = 0.22,
    window: int = 45,
    min_segment_frac: float = 0.08,
) -> list[tuple[int, int]]:
    """
    Делит диапазон [x1, x2) на под-диапазоны по резкой смене цвета —
    нужно, когда товары РАЗНЫХ брендов стоят вплотную без зазора, и
    проекция границ (см. _find_occupied_ranges) видит их как один общий
    блок. Между разными брендами обычно виден заметный скачок цвета
    (например, красная Coca-Cola -> зелёный Sprite), даже если физически
    бутылки соприкасаются.

    Если внутри диапазона цвет не меняется существенно (например, это
    просто несколько одинаковых бутылок одного бренда подряд), функция
    вернёт исходный диапазон без изменений — разбиение по цвету
    безопасно применять всегда, не только к заведомо широким блокам.

    Работает по HSV: берёт узкую горизонтальную полосу в средней трети
    диапазона по высоте (обычно это область этикетки) и для каждого
    столбца считает три признака — Hue (взвешенный по насыщенности,
    чтобы не сбивали блики/тени), Saturation и Value. Используются все
    три вместе (не только Hue): переход к тёмной/чёрной упаковке
    (например, Schweppes) почти не меняет Hue (у чёрного цвета оттенок
    не определён), зато резко проседает Saturation и Value — одного
    Hue для таких случаев недостаточно.
    """
    segment = image[:, x1:x2]
    h = segment.shape[0]
    if h < 10 or (x2 - x1) < window * 2:
        return [(x1, x2)]

    hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
    strip = hsv[int(h * 0.35) : int(h * 0.65), :, :].astype(np.float32)
    hue, sat, val = strip[:, :, 0], strip[:, :, 1], strip[:, :, 2]

    weighted_hue = (hue * sat).sum(axis=0) / np.maximum(sat.sum(axis=0), 1.0)
    mean_sat = sat.mean(axis=0)
    mean_val = val.mean(axis=0)

    w = len(weighted_hue)
    if w < window * 2:
        return [(x1, x2)]

    def smooth(arr, k=15):
        return np.convolve(arr, np.ones(k) / k, mode="same")

    # Нормализуем каждый признак в диапазон ~0..1, чтобы ни один из них
    # не доминировал в сравнении только из-за масштаба значений.
    feat_hue = smooth(weighted_hue) / 180.0
    feat_sat = smooth(mean_sat) / 255.0
    feat_val = smooth(mean_val) / 255.0
    features = np.stack([feat_hue, feat_sat, feat_val], axis=1)

    diff = np.zeros(w)
    for x in range(window, w - window):
        left = np.median(features[x - window : x], axis=0)
        right = np.median(features[x : x + window], axis=0)
        diff[x] = float(np.linalg.norm(right - left))

    candidates = np.where(diff > diff_threshold)[0]
    splits: list[int] = []
    min_gap = window
    for c in candidates:
        if not splits or c - splits[-1] > min_gap:
            splits.append(int(c))
        elif diff[c] > diff[splits[-1]]:
            splits[-1] = int(c)

    if not splits:
        return [(x1, x2)]

    boundaries = [0] + splits + [w]
    min_len = min_segment_frac * w
    segments = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        if seg_end - seg_start < min_len:
            # Слишком узкий кусок (часто краевой шум) — приклеиваем к
            # соседнему сегменту, а не оставляем как отдельный товар.
            if segments:
                segments[-1] = (segments[-1][0], seg_end)
            else:
                boundaries[i + 1] = boundaries[i]
            continue
        segments.append((seg_start, seg_end))

    if not segments:
        return [(x1, x2)]

    return [(x1 + s, x1 + e) for s, e in segments]


def _estimate_count(width: int, height: int) -> int:
    """
    Приблизительно оценивает, сколько отдельных единиц товара умещается
    в блоке заданной ширины, если предположить, что все они одного
    типоразмера (см. _TYPICAL_ITEM_ASPECT). Используется только когда
    товары стоят вплотную и разделить их поштучно не удалось — см.
    docstring модуля.
    """
    if height <= 0:
        return 1
    single_item_width = height * _TYPICAL_ITEM_ASPECT
    if single_item_width <= 0:
        return 1
    return max(1, round(width / single_item_width))


def detect_products(
    image: np.ndarray,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.8,
) -> list[BBox]:
    """
    Находит прямоугольные области отдельных товаров (или групп товаров
    одного цвета, если они стоят вплотную — см. _split_by_color) на фото
    ОДНОЙ полки.

    Алгоритм:
      1. Canny с умеренными порогами — здесь важно поймать любые следы
         объекта (включая слабоконтрастные), а не выделить один чистый
         замкнутый контур.
      2. Удаление длинных горизонтальных линий (кромка полки, ценники).
      3. Проекция границ по столбцам -> находим X-диапазоны товаров
         (промежутки между товарами почти всегда видны как явный провал).
      4. Доразбиение каждого X-диапазона по резкой смене цвета — на
         случай, если внутри него на самом деле несколько разных
         брендов, стоящих вплотную (см. _split_by_color).
      5. Для каждого итогового X-диапазона — проекция по строкам внутри
         него -> находим полную высоту товара.
      6. Оценка количества единиц товара в получившейся рамке по её
         ширине (см. _estimate_count) — обычно 1, кроме случая плотно
         стоящих товаров без зазора.

    min_area_ratio / max_area_ratio — доля площади кадра, которую может
    занимать один товар; отсекают явный шум и артефакты после проекции.

    Возвращает список bounding box-ов в пиксельных координатах.
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)

    horiz_min_length = max(60, int(w_img * 0.6))
    edges_clean = _remove_long_horizontal_lines(edges, horiz_min_length)

    col_profile = (edges_clean > 0).sum(axis=0).astype(np.float32)
    col_ranges = _find_occupied_ranges(
        col_profile, thresh_frac=0.12, min_len_frac=0.03, gap_merge=5
    )

    # Доразбиваем по цвету — безопасно для уже нормально разделённых
    # одиночных товаров (см. docstring _split_by_color), но критично для
    # блоков из нескольких брендов, стоящих вплотную без зазора.
    color_split_ranges: list[tuple[int, int]] = []
    for x1, x2 in col_ranges:
        color_split_ranges.extend(_split_by_color(image, x1, x2))

    boxes: list[BBox] = []
    for x1, x2 in color_split_ranges:
        sub = edges_clean[:, x1:x2]
        row_profile = (sub > 0).sum(axis=1).astype(np.float32)
        row_ranges = _find_occupied_ranges(
            row_profile, thresh_frac=0.15, min_len_frac=0.05, gap_merge=60
        )
        for y1, y2 in row_ranges:
            w, h = x2 - x1, y2 - y1
            ratio = (w * h) / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            # Слишком низкая рамка почти наверняка текст/ценник, а не
            # товар — это надёжнее, чем ограничивать соотношение сторон:
            # после разбиения по цвету (см. _split_by_color) один блок
            # может законно содержать несколько товаров подряд и быть
            # значительно шире одной высоты.
            if h < 0.08 * h_img:
                continue
            aspect = w / h if h else 0
            if aspect < 0.1 or aspect > 6.0:
                continue
            count = _estimate_count(w, h)
            boxes.append(BBox(x1, y1, w, h, count=count))

    return boxes


def auto_crop(image: np.ndarray, sharpness_thresh_frac: float = 0.15) -> tuple[int, int, int, int]:
    """
    Пытается автоматически найти область с самим прилавком, чтобы
    исключить фон — соседние стеллажи, проход между рядами и т.п.

    Работает на принципе, часто встречающемся в товарных фото: сам
    прилавок снят в фокусе, а окружение (соседние ряды магазина)
    намеренно размыто (боке), чтобы не отвлекать внимание. Резкость
    (дисперсия оператора Лапласа) в зоне фокуса заметно выше, чем в
    размытом фоне — по разнице и находим границы.

    Если чёткой границы не находится (например, всё фото одинаково
    резкое — реальное фото мерчендайзера с телефона без боке), честно
    возвращает границы всего изображения без изменений: автообрезка
    рассчитана как попытка "лучше, чем ничего", а не как гарантированно
    надёжный способ — при малейшем сомнении лучше не резать вовсе, чем
    отрезать часть настоящего товара.

    Возвращает (x1, y1, x2, y2) — необязательно совпадает с логотипом
    самого прилавка идеально, но обычно достаточно, чтобы убрать фон.
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    # По X: резкость считаем по средней трети кадра по высоте — там
    # обычно сама полка с товарами, а не пол/потолок.
    band = lap[h_img // 3 : 2 * h_img // 3, :]
    col_sharpness = band.var(axis=0)
    col_smooth = np.convolve(col_sharpness, np.ones(21) / 21, mode="same")

    col_thresh = sharpness_thresh_frac * col_smooth.max()
    x_focus = np.where(col_smooth > col_thresh)[0]

    if len(x_focus) < 0.2 * w_img:
        # Слишком мало "резких" столбцов — похоже, разделения на фокус/
        # фон в этом фото просто нет. Не рискуем резать наугад.
        x1, x2 = 0, w_img
    else:
        x1, x2 = int(x_focus.min()), int(x_focus.max())

    # По Y — аналогично, но по уже найденному X-диапазону, чтобы не
    # путать эффект от фона по бокам.
    row_band = lap[:, x1:x2] if x2 > x1 else lap
    row_sharpness = row_band.var(axis=1)
    row_smooth = np.convolve(row_sharpness, np.ones(15) / 15, mode="same")
    row_thresh = sharpness_thresh_frac * row_smooth.max()
    y_focus = np.where(row_smooth > row_thresh)[0]

    if len(y_focus) < 0.2 * h_img:
        y1, y2 = 0, h_img
    else:
        y1, y2 = int(y_focus.min()), int(y_focus.max())

    return x1, y1, x2, y2