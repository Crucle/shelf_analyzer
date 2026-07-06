"""
Анализатор прилавков — веб-версия на Streamlit.

Локальный запуск:
    streamlit run app.py

Деплой на Streamlit Community Cloud: см. README.md, раздел
"Публикация на Streamlit Community Cloud".

Распознавание бренда товара использует мультимодальную модель CLIP
(см. brand_classifier.py) — при первом запуске модель скачивается из
интернета (~600 МБ вместе с зависимостями), дальше работает офлайн.
"""
import cv2
import numpy as np
import streamlit as st

from brand_classifier import classify_brands
from detector import auto_crop
from layout_checker import check_layout
from main import DEFAULT_BRANDS, _detect_all_rows
from report import build_layout_report
from visualizer import draw_result

st.set_page_config(page_title="Анализатор прилавков", page_icon="🛒", layout="wide")


def load_image(uploaded_file) -> np.ndarray:
    """Декодирует загруженный в браузере файл в изображение OpenCV (BGR)."""
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Загружаю мультимодальную модель CLIP (один раз)...")
def _warmup_model():
    """
    Модель CLIP грузится с диска/интернета один раз за время жизни
    приложения (st.cache_resource) — иначе она загружалась бы заново на
    каждый клик "Анализировать", что было бы очень медленно.
    """
    from brand_classifier import _get_model

    _get_model()
    return True


def main():
    st.title("🛒 Анализатор прилавков")
    st.caption(
        "Проверка корректности выкладки товаров по фото: находит отдельные "
        "товары, распознаёт бренд с помощью мультимодальной модели CLIP и "
        "определяет, стоят ли одинаковые товары единым блоком, или разбиты "
        "другими товарами."
    )

    uploaded_file = st.file_uploader(
        "Загрузите фото прилавка", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Загрузите фото, чтобы начать анализ.")
        return

    image = load_image(uploaded_file)
    if image is None:
        st.error("Не удалось открыть изображение. Попробуйте другой файл.")
        return

    img_h, img_w = image.shape[:2]

    # Автообрезка (см. detector.auto_crop) считается один раз для
    # каждого нового загруженного фото и служит стартовым значением
    # ползунков — дальше пользователь может подправить их вручную, если
    # автоматика ошиблась. Пересчитываем только когда меняется само фото
    # (а не при каждом перерисовывании страницы).
    file_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("_crop_file_key") != file_key:
        st.session_state["_crop_file_key"] = file_key
        ax1, ay1, ax2, ay2 = auto_crop(image)
        st.session_state["_crop_x"] = (ax1, ax2)
        st.session_state["_crop_y"] = (ay1, ay2)

    with st.sidebar:
        st.header("Настройки")

        st.subheader("Обрезка кадра")
        st.caption(
            "Ползунки уже настроены автоматически (по резкости — сам "
            "прилавок обычно в фокусе, а фон вокруг размыт). Если "
            "автообрезка ошиблась — поправьте вручную; кнопка ниже "
            "сбрасывает на всё фото целиком."
        )
        if st.button("Сбросить на всё фото"):
            st.session_state["_crop_x"] = (0, img_w)
            st.session_state["_crop_y"] = (0, img_h)
        x1, x2 = st.slider("Диапазон по X", 0, img_w, key="_crop_x")
        y1, y2 = st.slider("Диапазон по Y", 0, img_h, key="_crop_y")

        st.subheader("Полки")
        rows = st.number_input(
            "Сколько физических полок на фото (по высоте)",
            min_value=1,
            max_value=10,
            value=1,
            help=(
                "Детектор надёжно работает с одной полкой за раз. Если на "
                "фото весь стеллаж — укажите реальное число полок, кадр "
                "поделится на столько же равных горизонтальных полос."
            ),
        )

        st.subheader("Бренды для распознавания")
        brands_text = st.text_area(
            "Список брендов через запятую",
            value=", ".join(DEFAULT_BRANDS.keys()),
            help=(
                "CLIP выберет для каждого товара наиболее подходящий бренд "
                "из этого списка. Впишите те бренды, которые реально есть "
                "на вашем фото — чем точнее список, тем точнее результат. "
                "Если оставить список по умолчанию без изменений, "
                "используются встроенные подробные описания каждого бренда "
                "(цвет крышки, этикетки и т.п.) — они точнее, чем просто "
                "название, особенно для похожих по цвету товаров."
            ),
        )
        brand_names_from_text = [b.strip() for b in brands_text.split(",") if b.strip()]
        if brand_names_from_text == list(DEFAULT_BRANDS.keys()):
            # Список не менялся — используем встроенные подробные описания
            # (с несколькими фразами на бренд), а не просто голые названия.
            brands = DEFAULT_BRANDS
        else:
            brands = brand_names_from_text

        min_confidence = st.slider(
            "Минимальная уверенность модели",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help=(
                "Если уверенность модели в лучшем бренде ниже этого порога, "
                'товар помечается как "не определено" вместо навязанного, '
                "но малоуверенного варианта."
            ),
        )

        st.subheader("Эталонные фото-примеры (по желанию)")
        st.caption(
            "Если модель путает похожие товары (например, две прозрачные "
            "бутылки воды разных марок) — покажите ей несколько реальных "
            'фото каждого бренда. Назовите файлы вида "Schweppes_1.jpg" — '
            "часть имени до первого подчёркивания должна точно совпадать "
            "с названием бренда из списка выше."
        )
        ref_files = st.file_uploader(
            "Эталонные фото (можно выбрать сразу несколько)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        reference_images = {}
        if ref_files:
            brand_names_current = list(brands.keys()) if isinstance(brands, dict) else brands
            unmatched = []
            for f in ref_files:
                brand_guess = f.name.rsplit(".", 1)[0].split("_")[0].strip()
                matched = next(
                    (b for b in brand_names_current if b.lower() == brand_guess.lower()), None
                )
                if matched:
                    reference_images.setdefault(matched, []).append(load_image(f))
                else:
                    unmatched.append(f.name)
            if unmatched:
                st.warning(
                    f"Не удалось определить бренд по имени файла: {', '.join(unmatched)}. "
                    'Переименуйте как "Бренд_1.jpg".'
                )
            if reference_images:
                st.success(
                    "Загружены эталоны: "
                    + ", ".join(f"{k} ({len(v)})" for k, v in reference_images.items())
                )

        run = st.button("Анализировать", type="primary", use_container_width=True)

    cropped = image[y1:y2, x1:x2]

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Исходное фото (после обрезки)")
        st.image(bgr_to_rgb(cropped), use_container_width=True)

    if not run:
        with col_right:
            st.info("Настройте параметры слева и нажмите «Анализировать».")
        return

    if not brands:
        st.error("Список брендов пуст — впишите хотя бы один бренд слева.")
        return

    _warmup_model()

    with st.spinner("Анализирую..."):
        boxes = _detect_all_rows(cropped, rows)
        if not boxes:
            labels, violations, matches = [], [], []
        else:
            matches = classify_brands(
                cropped,
                boxes,
                brands,
                reference_images=reference_images or None,
                min_confidence=min_confidence,
            )
            labels = [m.brand for m in matches]
            violations = check_layout(boxes, labels)

    with col_right:
        st.subheader("Результат")
        if not boxes:
            st.warning(
                "Товары не найдены. Попробуйте изменить область обрезки — "
                "возможно, в кадре мало контраста между товарами и фоном."
            )
        else:
            violation_labels = {v.label for v in violations}
            result_img = draw_result(cropped, boxes, labels, violation_labels)
            st.image(bgr_to_rgb(result_img), use_container_width=True)

    if not boxes:
        return

    image_area = cropped.shape[0] * cropped.shape[1]
    report = build_layout_report(image_area, boxes, labels, violations)

    st.divider()
    st.header("📋 Отчёт о выкладке")

    for w in report["warnings"]:
        st.warning(f"⚠️ {w}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Соответствие выкладке", f"{report['compliance_percent']}%")
    m2.metric("Критических нарушений", report["critical_violations_count"])
    m3.metric("Полок распознано", report["shelves_recognized"])
    m4.metric(
        "Товаров найдено",
        report["total_products"],
        delta="оценка" if report["count_is_estimated"] else "точно",
        delta_color="off",
    )

    st.progress(report["compliance_percent"] / 100)

    st.write(report["summary"])

    if report["critical_violations"]:
        st.subheader("Критические нарушения")
        for v in report["critical_violations"]:
            st.error(v)
    else:
        st.success("Нарушений выкладки не найдено — одинаковые товары стоят блоками.")

    with st.expander("Уверенность модели по каждому товару"):
        for m in matches:
            st.write(f"- {m.brand}: {m.confidence:.0%}")

    with st.expander("Полный отчёт (JSON)"):
        st.json(report)


if __name__ == "__main__":
    main()
