from dataclasses import dataclass

import numpy as np

from detector import BBox


@dataclass
class Violation:
    label: str
    row: int
    message: str


def group_into_rows(boxes: list[BBox], row_tolerance: float = 0.6) -> list[list[int]]:
    if not boxes:
        return []

    avg_h = float(np.mean([b.h for b in boxes]))

    order_by_y = sorted(range(len(boxes)), key=lambda i: boxes[i].center[1])
    rows: list[list[int]] = []
    for idx in order_by_y:
        y_c = boxes[idx].center[1]
        placed = False
        for row in rows:
            row_y = np.mean([boxes[j].center[1] for j in row])
            if abs(y_c - row_y) < avg_h * row_tolerance:
                row.append(idx)
                placed = True
                break
        if not placed:
            rows.append([idx])

    for row in rows:
        row.sort(key=lambda i: boxes[i].center[0])
    rows.sort(key=lambda row: np.mean([boxes[j].center[1] for j in row]))
    return rows


def check_layout(boxes: list[BBox], labels: list) -> list[Violation]:
    violations: list[Violation] = []
    rows = group_into_rows(boxes)

    for row_num, row in enumerate(rows):
        closed_labels: set = set()
        current_label = None
        for idx in row:
            label = labels[idx]
            if label != current_label:
                if current_label is not None:
                    closed_labels.add(current_label)
                current_label = label
            if label in closed_labels:
                violations.append(
                    Violation(
                        label=label,
                        row=row_num,
                        message=(
                            f'Товар "{label}" в строке {row_num} '
                            f"разбит на несколько блоков вместо одного."
                        ),
                    )
                )
    return violations
