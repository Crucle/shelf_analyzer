import difflib
import json
from dataclasses import dataclass, field


@dataclass
class ShelfDiff:
    shelf_number: int           
    expected: list               
    actual: list                 
    positions: list = field(default_factory=list)  
    missing: list = field(default_factory=list)    
    extra: list = field(default_factory=list)       


def load_planogram(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_planogram(planogram: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(planogram, f, ensure_ascii=False, indent=2)


def _positional_diff(expected: list, actual: list) -> list:
    sm = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    result = []
    position = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                position += 1
                result.append(
                    {"position": position, "status": "correct", "expected": expected[i1 + k], "actual": actual[j1 + k]}
                )
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                position += 1
                exp = expected[i1 + k] if k < (i2 - i1) else None
                act = actual[j1 + k] if k < (j2 - j1) else None
                if exp is None:
                    result.append({"position": position, "status": "extra", "expected": None, "actual": act})
                elif act is None:
                    result.append({"position": position, "status": "missing", "expected": exp, "actual": None})
                else:
                    result.append({"position": position, "status": "wrong_item", "expected": exp, "actual": act})
        elif tag == "delete":
            for k in range(i2 - i1):
                position += 1
                result.append({"position": position, "status": "missing", "expected": expected[i1 + k], "actual": None})
        elif tag == "insert":
            for k in range(j2 - j1):
                position += 1
                result.append({"position": position, "status": "extra", "expected": None, "actual": actual[j1 + k]})
    return result


def compare_to_planogram(actual_rows: list, planogram: list) -> list:
    diffs = []
    num_shelves = max(len(planogram), len(actual_rows))
    for i in range(num_shelves):
        expected = planogram[i] if i < len(planogram) else []
        actual = actual_rows[i] if i < len(actual_rows) else []
        positions = _positional_diff(expected, actual)
        missing = [p["expected"] for p in positions if p["status"] == "missing"]
        extra = [p["actual"] for p in positions if p["status"] == "extra"]
        diffs.append(
            ShelfDiff(
                shelf_number=i + 1,
                expected=expected,
                actual=actual,
                positions=positions,
                missing=missing,
                extra=extra,
            )
        )
    return diffs


def violations_as_messages(diffs: list) -> list:
    messages = []
    for d in diffs:
        for p in d.positions:
            if p["status"] == "wrong_item":
                messages.append(
                    f'Полка {d.shelf_number}, место {p["position"]}: должен стоять '
                    f'"{p["expected"]}", а фактически стоит "{p["actual"]}"'
                )
            elif p["status"] == "missing":
                messages.append(
                    f'Полка {d.shelf_number}, место {p["position"]}: отсутствует товар "{p["expected"]}"'
                )
            elif p["status"] == "extra":
                messages.append(
                    f'Полка {d.shelf_number}, место {p["position"]}: лишний товар "{p["actual"]}" '
                    f'— по планограмме здесь его быть не должно'
                )
    return messages


def summarize(diffs: list) -> dict:
    all_positions = [p for d in diffs for p in d.positions]
    total_expected = sum(len(d.expected) for d in diffs)

    correct = sum(1 for p in all_positions if p["status"] == "correct")
    wrong_item = sum(1 for p in all_positions if p["status"] == "wrong_item")
    total_missing = sum(1 for p in all_positions if p["status"] == "missing")
    total_extra = sum(1 for p in all_positions if p["status"] == "extra")

    match_percent = round(100 * correct / total_expected) if total_expected else 100

    return {
        "match_percent": match_percent,
        "total_expected": total_expected,
        "correct_positions": correct,
        "wrong_item_positions": wrong_item,
        "total_missing": total_missing,
        "total_extra": total_extra,
        "is_fully_correct": wrong_item == 0 and total_missing == 0 and total_extra == 0,
    }
