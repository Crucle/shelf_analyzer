"""
Планограмма — эталонная схема выкладки: для каждой полки задан список
брендов в том порядке, в котором они должны стоять на этой полке слева
направо (одна позиция в списке = один физический товар на полке).

Сравнение фактической выкладки (того, что программа нашла и распознала
на фото) с планограммой позволяет находить три независимых вида
нарушений на каждой полке:
  - отсутствующие товары — есть в планограмме, но не найдены на фото;
  - лишние товары — найдены на фото, но их не должно быть по планограмме;
  - неправильный порядок — все нужные товары на месте (или почти), но
    расположены не в том порядке, что предусмотрено планограммой.

Формат планограммы — список списков названий брендов, один список на
полку, сверху вниз: [["Coca-Cola", "Coca-Cola", "Sprite"], ["Fanta", ...]].
Хранится и передаётся как обычный JSON, чтобы планограмму можно было
один раз составить и переиспользовать для разных фото одного и того же
прилавка.
"""
import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ShelfDiff:
    shelf_number: int          # номер полки, считая с 1 (сверху)
    expected: list             # ожидаемая последовательность брендов
    actual: list                # фактически найденная последовательность
    missing: list = field(default_factory=list)   # чего не хватает
    extra: list = field(default_factory=list)      # что лишнее
    order_correct: bool = True  # совпадает ли порядок ОБЩИХ товаров


def load_planogram(path: str) -> list:
    """Загружает планограмму из JSON-файла (список списков брендов)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_planogram(planogram: list, path: str) -> None:
    """Сохраняет планограмму в JSON-файл — чтобы переиспользовать её позже."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(planogram, f, ensure_ascii=False, indent=2)


def _diff_counts(expected: list, actual: list) -> tuple[list, list]:
    """
    Сравнивает два списка как мультимножества (учитывая количество
    повторов, а не только сам факт наличия бренда) и возвращает
    (missing, extra).
    """
    exp_counter = Counter(expected)
    act_counter = Counter(actual)
    missing = list((exp_counter - act_counter).elements())
    extra = list((act_counter - exp_counter).elements())
    return missing, extra


def _filter_to_common(seq: list, common: Counter) -> list:
    """Оставляет в последовательности только те элементы, что входят в common (с учётом количества)."""
    remaining = Counter(common)
    result = []
    for item in seq:
        if remaining.get(item, 0) > 0:
            result.append(item)
            remaining[item] -= 1
    return result


def _check_order(expected: list, actual: list) -> bool:
    """
    Проверяет порядок ОБЩИХ товаров: сначала убираем из обеих
    последовательностей всё то, чего нет в другой (лишнее/недостающее —
    это отдельная, уже посчитанная проблема), а затем сравниваем
    оставшееся один-в-один. Так неправильный порядок фиксируется
    отдельно от простого несовпадения набора товаров.
    """
    common = Counter(expected) & Counter(actual)
    return _filter_to_common(expected, common) == _filter_to_common(actual, common)


def compare_to_planogram(actual_rows: list, planogram: list) -> list:
    """
    Сравнивает фактически распознанные товары по полкам (actual_rows —
    список списков брендов, по одному списку на полку, слева направо, в
    том же порядке полок, что и планограмма — сверху вниз) с
    планограммой. Возвращает список ShelfDiff, по одному на каждую полку.

    Если полок в планограмме и на фото не поровну, недостающие полки
    считаются полностью пустыми (все товары на них — "отсутствующие"),
    а лишние полки на фото — полностью "лишними".
    """
    diffs = []
    num_shelves = max(len(planogram), len(actual_rows))
    for i in range(num_shelves):
        expected = planogram[i] if i < len(planogram) else []
        actual = actual_rows[i] if i < len(actual_rows) else []
        missing, extra = _diff_counts(expected, actual)
        order_correct = _check_order(expected, actual)
        diffs.append(
            ShelfDiff(
                shelf_number=i + 1,
                expected=expected,
                actual=actual,
                missing=missing,
                extra=extra,
                order_correct=order_correct,
            )
        )
    return diffs


def summarize(diffs: list) -> dict:
    """
    Сводка по всем полкам сразу: сколько всего отсутствует/лишнее,
    сколько полок с неправильным порядком, и общий процент соответствия
    (доля товаров из планограммы, которые нашлись на своих местах).
    """
    total_expected = sum(len(d.expected) for d in diffs)
    total_missing = sum(len(d.missing) for d in diffs)
    total_extra = sum(len(d.extra) for d in diffs)
    shelves_with_order_issues = sum(1 for d in diffs if not d.order_correct and not d.missing and not d.extra)

    matched = total_expected - total_missing
    match_percent = round(100 * matched / total_expected) if total_expected else 100

    return {
        "match_percent": match_percent,
        "total_expected": total_expected,
        "total_missing": total_missing,
        "total_extra": total_extra,
        "shelves_with_order_issues": shelves_with_order_issues,
        "is_fully_correct": total_missing == 0 and total_extra == 0 and shelves_with_order_issues == 0,
    }
