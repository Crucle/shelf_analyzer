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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_planogram(planogram: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(planogram, f, ensure_ascii=False, indent=2)


def _diff_counts(expected: list, actual: list) -> tuple[list, list]:
    exp_counter = Counter(expected)
    act_counter = Counter(actual)
    missing = list((exp_counter - act_counter).elements())
    extra = list((act_counter - exp_counter).elements())
    return missing, extra


def _filter_to_common(seq: list, common: Counter) -> list:
    remaining = Counter(common)
    result = []
    for item in seq:
        if remaining.get(item, 0) > 0:
            result.append(item)
            remaining[item] -= 1
    return result


def _check_order(expected: list, actual: list) -> bool:
    common = Counter(expected) & Counter(actual)
    return _filter_to_common(expected, common) == _filter_to_common(actual, common)


def compare_to_planogram(actual_rows: list, planogram: list) -> list:
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
