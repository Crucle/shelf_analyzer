"""
Сборка итогового отчёта о выкладке в формате дашборда: процент
соответствия, число критических нарушений, число распознанных полок,
текстовое резюме и список конкретных нарушений.

У нас нет эталонной планограммы (плана "что где должно стоять") — вместо
сравнения с ней используется собственная метрика на основе фактически
найденных нарушений (см. compliance_percent ниже). Если планограмма
появится, эту метрику легко заменить на честное сравнение факт/план —
интерфейс функции (набор входных данных → тот же набор полей отчёта)
менять не придётся.
"""
from detector import BBox
from layout_checker import Violation, group_into_rows


def _compliance_percent(brands_found: list[str], violations: list[Violation]) -> int:
    """
    Доля брендов, у которых НЕ было найдено нарушений (стоят единым
    блоком), от всех найденных брендов. 100% — все бренды разложены
    правильно, 0% — у каждого найденного бренда есть хотя бы один разрыв.
    """
    if not brands_found:
        return 100
    brands_with_violation = {v.label for v in violations}
    ok_brands = [b for b in brands_found if b not in brands_with_violation]
    return round(100 * len(ok_brands) / len(brands_found))


def _compliance_comment(percent: int) -> str:
    if percent >= 90:
        return "выкладка практически полностью корректна"
    if percent >= 60:
        return "есть отдельные нарушения выкладки"
    if percent >= 30:
        return "заметная часть выкладки нарушена"
    return "выкладка нарушена почти полностью"


def _find_oversized_boxes(boxes: list[BBox], image_area: int, ratio_threshold: float = 0.15) -> list[BBox]:
    """
    Рамка, где площадь В РАСЧЁТЕ НА ОДНУ ОЦЕНЁННУЮ ЕДИНИЦУ ТОВАРА (см.
    BBox.count в detector.py) занимает слишком большую долю кадра, почти
    всегда означает, что в кадр попал лишний фон (соседний стеллаж, часть
    прохода) и детектор поймал его как один "товар". Используется, чтобы
    предупредить пользователя — см. warnings в build_layout_report.

    Считаем именно на одну единицу, а не на весь бокс целиком: блок из
    нескольких товаров, стоящих вплотную (см. _split_by_color в
    detector.py), законно может занимать значительную часть кадра — это
    не признак лишнего фона.
    """
    result = []
    for b in boxes:
        per_item_ratio = (b.area / max(b.count, 1)) / image_area if image_area else 0
        if per_item_ratio > ratio_threshold:
            result.append(b)
    return result


def build_layout_report(
    image_area: int,
    boxes: list[BBox],
    labels: list[str],
    violations: list[Violation],
) -> dict:
    """
    Собирает итоговый отчёт. image_area нужен только для проверки на
    "слишком большие" рамки (см. _find_oversized_boxes) — на сам расчёт
    соответствия не влияет.
    """
    brands_found = sorted(set(labels)) if labels else []
    shelves_recognized = len(group_into_rows(boxes))
    compliance = _compliance_percent(brands_found, violations)

    total_products = sum(b.count for b in boxes)
    positions_detected = len(boxes)
    has_estimated = any(b.count > 1 for b in boxes)

    warnings: list[str] = []
    oversized = _find_oversized_boxes(boxes, image_area)
    if oversized:
        warnings.append(
            f"Найдено {len(oversized)} подозрительно крупных рамок — "
            "вероятно, в кадр попал лишний фон (соседний стеллаж, проход "
            "между рядами). Обрежьте фото через --crop (или в веб-версии "
            "— ползунками обрезки), чтобы убрать его из анализа."
        )
    if has_estimated:
        warnings.append(
            "Некоторые товары стоят вплотную друг к другу без видимого "
            "промежутка — точное количество внутри такой группы не "
            "подсчитывается поштучно, а оценивается приблизительно по "
            "ширине группы (см. поле total_products — это оценка, а не "
            "точный подсчёт)."
        )

    if not boxes:
        summary = "Товары на фото не найдены — проверьте область обрезки и контраст между товарами и фоном."
    else:
        summary = (
            f"На фото распознано полок: {shelves_recognized}. "
            f"Всего найдено товаров: {total_products}"
            f"{' (оценочно)' if has_estimated else ''}, представлены бренды: "
            f"{', '.join(brands_found)}. Соответствие выкладки оценено в "
            f"{compliance}% — {_compliance_comment(compliance)}. "
            f"Критических нарушений: {len(violations)}."
        )

    return {
        "compliance_percent": compliance,
        "critical_violations_count": len(violations),
        "shelves_recognized": shelves_recognized,
        "total_products": total_products,
        "positions_detected": positions_detected,
        "count_is_estimated": has_estimated,
        "brands_found": brands_found,
        "summary": summary,
        "critical_violations": [v.message for v in violations],
        "warnings": warnings,
    }