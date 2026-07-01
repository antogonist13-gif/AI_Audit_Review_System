from __future__ import annotations

import sys
import textwrap
import html
from pathlib import Path
import tempfile

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.svk_analytics.columns import build_canonical_frame, resolve_columns
from src.svk_analytics.io import find_latest_raw_file, load_report, load_yaml
from src.svk_analytics.scoring import _form_name, enrich_with_metrics
from src.svk_analytics.summaries import (
    anomalies_table,
    by_dimension_summary,
    contradictions_table,
    data_quality_issues,
    direction_summary,
    effectiveness_review_summary,
    improvement_actions_summary,
    legal_responsibility_measures,
    management_actions,
    normalized_activity_metrics,
    overview_metrics,
    cloud_peer_benchmark,
    cloud_peer_benchmark_table,
    get_cloud_peer_neighbor_names,
    proportionality_anomalies,
    scale_axis_profile,
    split_extreme_values,
    split_profile_outliers,
    report_status_summary,
    risk_group_summary,
    risk_methodology_summary,
    svk_elements_summary,
    svk_form_flags_summary,
    svk_form_level_activity_summary,
    svk_form_level_summary,
    top_risk_organizations,
    violations_and_remediation,
    violations_summary,
)

st.set_page_config(
    page_title="Мониторинг организации и осуществления внутреннего контроля организаций, подведомственных Минобрнауки России",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_canonical_report(path: str, file_mtime: float | None = None):
    """Кэшируется только чтение и нормализация отчёта; расчёт метрик — всегда свежий."""
    columns_config = load_yaml("config/columns.yml")
    raw = load_report(path)
    resolved, resolution_report = resolve_columns(raw, columns_config)
    canonical = build_canonical_frame(raw, resolved)
    return canonical, resolution_report


def load_and_score(path: str, year: int | None, min_group_size: int | None = None):
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else None
    canonical, resolution_report = load_canonical_report(path, mtime)
    scoring_config = load_yaml("config/scoring.yml")
    if min_group_size is not None:
        scoring_config.setdefault("peer_benchmark", {})["min_group_size"] = min_group_size
    enriched = enrich_with_metrics(canonical, scoring_config, year=year)
    return enriched, scoring_config, resolution_report


def pct(x):
    """Format percentage value. Accepts either 0-1 or 0-100 ranges."""
    if pd.isna(x):
        return "—"
    # If value is between 0 and 1, treat as proportion and convert to percentage
    if 0 <= x <= 1:
        return f"{x * 100:.1f}%"
    # If value is greater than 1, treat as already in percentage
    return f"{x:.1f}%"


def fmt_num(x, digits: int = 0):
    if x is None or pd.isna(x):
        return "—"
    if digits == 0:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:,.{digits}f}".replace(",", " ")


def value_from_overview(overview: pd.DataFrame, metric: str):
    if overview.empty or metric not in set(overview["metric"]):
        return None
    return overview.loc[overview["metric"] == metric, "value"].iloc[0]


def wrap_plain_text(value, width: int = 34) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def metric_card(label: str, value):
    label_html = "<br>".join(
        textwrap.wrap(
            html.escape(str(label)),
            width=28,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    value_html = html.escape(str(value))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label_html}</div>
            <div class="metric-card-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DISPLAY_LABELS = {
    "% заполненных отчетов": "Доля заполненных отчётов",
    "% организаций, где СВК организован": "Доля организаций, где внутренний контроль организован и осуществляется",
    "% организаций с полным базовым комплектом СВК": "Доля организаций, где организация внутреннего контроля полностью формализована",
    "% организаций с полноценной методикой оценки рисков": "Доля организаций, где методика по выявлению и оценке рисков применяется",
    "% организаций с регулярной оценкой эффективности СВК": "Доля организаций с регулярной оценкой эффективности организации и осуществления внутреннего контроля",
    "Выявленные нарушения": "Количество выявленных нарушений, ед.",
    "% нарушений, устраненных в срок": "Доля нарушений, устранённых в срок",
    "Организаций, где СВК организован": "Количество организаций, где внутренний контроль организован и осуществляется",
    "Организаций с полным базовым комплектом СВК": "Количество организаций, где организация внутреннего контроля полностью формализована",
    "Организаций с регулярной оценкой эффективности СВК": "Количество организаций с регулярной оценкой эффективности организации и осуществления внутреннего контроля",
    "Организаций со слабой формой СВК": "Количество организаций, где форма обеспечения функционирования внутреннего контроля ниже расчётной рекомендации",
    "СВК организован": "Внутренний контроль организован и осуществляется",
    "СВК организован и осуществляется": "Внутренний контроль организован и осуществляется",
    "Раздел СВК в учетной политике": "Раздел/положение об организации и осуществлении внутреннего контроля в учётной политике",
    "Организация СВК утверждена ЛНА": "Организация и осуществление внутреннего контроля утверждена ЛНА",
    "СВК утвержден ЛНА": "Организация и осуществление внутреннего контроля утверждена ЛНА",
    "Определены полномочия": "Определены полномочия ответственных лиц",
    "План-график мероприятий СВК": "Утверждённый план-график мероприятий по организации и осуществлению внутреннего контроля",
    "План-график мероприятий": "Утверждённый план-график мероприятий по организации и осуществлению внутреннего контроля",
    "Уполномоченное должностное лицо": "Уполномоченное должностное лицо",
    "Назначено уполномоченное должностное лицо": "Уполномоченное должностное лицо",
    "Временный коллегиальный орган": "Временный коллегиальный орган",
    "Временный коллегиальный орган / комиссия": "Временный коллегиальный орган",
    "Постоянный коллегиальный орган": "Постоянный коллегиальный орган",
    "Постоянно действующий коллегиальный орган": "Постоянный коллегиальный орган",
    "Структурное подразделение": "Уполномоченное структурное подразделение",
    "Уполномоченное структурное подразделение": "Уполномоченное структурное подразделение",
    "Иная форма": "Иная форма",
    "Иное": "Иная форма",
    "ФХД": "Финансово-хозяйственная деятельность",
    "Закупки": "Деятельность по закупкам",
    "Имущество": "Использование и распоряжение имуществом",
    "Проекты": "Проектная деятельность",
    "Количество выявленных нарушений": "Количество выявленных нарушений, ед.",
    "Устранено в срок": "Из них устранено в срок, ед.",
    "Устранено с нарушением срока": "Из них устранено с нарушением срока, ед.",
    "Остаток без отраженного устранения": "Остаток без отражённого устранения, ед.",
    "Планы устранения": "Количество разработанных планов устранения, ед.",
    "Изменения в ЛНА": "Количество принятых решений по внесению изменений в ЛНА, ед.",
    "Дисциплинарные решения": "Количество решений о привлечении к дисциплинарной ответственности, ед.",
    "Дисциплинарная ответственность: решений принято": "Количество решений о привлечении к дисциплинарной ответственности, ед.",
    "  → из них отменены в порядке обжалования": "Из них отменены в порядке обжалования",
    "Отменены в порядке обжалования": "Из них отменены в порядке обжалования",
    "Материалы в правоохранительные органы": "Количество материалов, направленных в правоохранительные органы и органы госконтроля и надзора, ед.",
    "Материалы в правоохранительные органы: направлено": "Количество материалов, направленных в правоохранительные органы и органы госконтроля и надзора, ед.",
    "  → из них возвращены, получен отказ": "Из них возвращены, получен отказ",
    "Возвращены / отказ": "Из них возвращены, получен отказ",
    "Материалы в суд": "Количество материалов, направленных в суд, ед.",
    "Материалы в суд: направлено": "Количество материалов, направленных в суд, ед.",
    "  → из них отказано в удовлетворении": "Из них отказано в удовлетворении",
    "Отказано в удовлетворении": "Из них отказано в удовлетворении",
    "Полноценная методика": "Методика по выявлению и оценке рисков применяется",
    "полноценная методика": "Методика по выявлению и оценке рисков применяется",
    "Фрагментарная методика": "Методика применяется фрагментарно",
    "фрагментарная / несистемная методика": "Методика применяется фрагментарно",
    "Методика отсутствует": "Методика оценки рисков отсутствует",
    "методика отсутствует": "Методика оценки рисков отсутствует",
    "Регулярная оценка": "Регулярная оценка эффективности внутреннего контроля проводится",
    "регулярная оценка": "Регулярная оценка эффективности внутреннего контроля проводится",
    "Нерегулярная оценка": "Оценка проводится нерегулярно или результаты не оформляются",
    "нерегулярная оценка / результаты не оформляются": "Оценка проводится нерегулярно или результаты не оформляются",
    "Оценка не проводится": "Оценка эффективности внутреннего контроля не проводится",
    "оценка не проводится": "Оценка эффективности внутреннего контроля не проводится",
    "Документированные мероприятия реализуются": "Мероприятия по совершенствованию внутреннего контроля реализуются",
    "документированные мероприятия реализуются": "Мероприятия по совершенствованию внутреннего контроля реализуются",
    "Мероприятия планируются или выполняются частично": "Мероприятия планируются или выполняются не в полном объёме",
    "мероприятия планируются или выполняются частично": "Мероприятия планируются или выполняются не в полном объёме",
    "Мероприятия не планируются": "Мероприятия не планируются и не проводятся",
    "мероприятия не планируются / не проводятся": "Мероприятия не планируются и не проводятся",
    "Покрыто": "Включено в контур внутреннего контроля",
    "Не покрыто": "Не включено в контур внутреннего контроля",
    "covered_orgs": "Включено в контур внутреннего контроля",
    "uncovered_orgs": "Не включено в контур внутреннего контроля",
    "A. Сбалансированная СВК": "Группа A — сбалансированный внутренний контроль (расчётный показатель)",
    "B. Недопокрытие активных направлений": "Группа B — активные направления контроля вне контура внутреннего контроля (расчётный показатель)",
    "C. Слабая форма СВК": "Группа C — форма обеспечения функционирования внутреннего контроля ниже расчётной рекомендации",
    "C. Слабая форма СВК при имеющейся нагрузке": "Группа C — форма обеспечения функционирования внутреннего контроля ниже расчётной рекомендации при имеющейся нагрузке",
    "D. Двойной риск": "Группа D — двойной расчётный риск",
    "D. Двойной риск: непокрытие + слабая форма": "Группа D — двойной расчётный риск",
    "E. Отчет не заполнен / данные требуют проверки": "Группа E — отчёт не заполнен / данные требуют проверки",
    "E. Отчёт не заполнен / данные требуют проверки": "Группа E — отчёт не заполнен / данные требуют проверки",
}

DISPLAY_TEXT_REPLACEMENTS = {
    "форма СВК": "форма обеспечения функционирования внутреннего контроля",
    "Форма СВК": "Форма обеспечения функционирования внутреннего контроля",
    "элементы СВК": "элементы организации и осуществления внутреннего контроля",
    "Элементы СВК": "Элементы организации и осуществления внутреннего контроля",
    "оценка СВК": "оценка эффективности организации и осуществления внутреннего контроля",
    "Оценка СВК": "Оценка эффективности организации и осуществления внутреннего контроля",
    "СВК": "внутренний контроль",
}


def display_label(value):
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    text = str(value)
    text = DISPLAY_LABELS.get(text, text)
    for src, dst in DISPLAY_TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def display_form_name(level: int) -> str:
    """User-facing form label; backend `_form_name(...)` text is not shown directly."""
    return display_label(_form_name(level))


def wrap_display_text(value, width: int = 34) -> str:
    label = display_label(value)
    if label is None:
        return label
    try:
        if pd.isna(label):
            return label
    except (TypeError, ValueError):
        pass
    return "<br>".join(
        textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False)
    )


def chart_title(value: str) -> str:
    return wrap_display_text(value, width=62)


def axis_title(value: str) -> str:
    return wrap_display_text(value, width=42)


def legend_title(value: str) -> str:
    return wrap_display_text(value, width=30)


def add_display_column(df: pd.DataFrame, column: str, display_column: str | None = None) -> pd.DataFrame:
    result = df.copy()
    if column in result.columns:
        out_col = display_column or f"{column}_display"
        result[out_col] = result[column].map(display_label)
    return result


def add_wrapped_display_column(
    df: pd.DataFrame,
    column: str,
    display_column: str | None = None,
    width: int = 34,
) -> pd.DataFrame:
    result = df.copy()
    if column in result.columns:
        out_col = display_column or f"{column}_display"
        result[out_col] = result[column].map(lambda value: wrap_display_text(value, width=width))
    return result


def with_display_labels(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].map(display_label)
    return result


def with_display_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(display_label)
    return result


OFFICIAL_TITLE = "Мониторинг организации и осуществления внутреннего контроля организаций, подведомственных Минобрнауки России"


st.title(OFFICIAL_TITLE)
st.caption("Аналитическая система по ежегодному сводному отчёту организаций, подведомственных Минобрнауки России")
st.markdown(
    """
    <style>
    .metric-card {
        min-height: 7rem;
        padding: 0.65rem 0.8rem 0.75rem 0.8rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.5rem;
        background: rgba(128, 128, 128, 0.06);
    }
    .metric-card-label {
        min-height: 3.6rem;
        color: inherit;
        opacity: 0.82;
        font-size: 0.875rem;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }
    .metric-card-value {
        color: inherit;
        font-size: 1.75rem;
        line-height: 1.2;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Источник данных")
    year = st.number_input("Год анализа", min_value=2020, max_value=2100, value=2025, step=1)
    uploaded = st.file_uploader("Загрузите отчёт мониторинга", type=["xls", "xlsx", "html", "htm"])

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".xls"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()
        input_path = tmp.name
        st.success(f"Загружен файл: {uploaded.name}")
    else:
        latest = find_latest_raw_file("data/raw")
        input_path = str(latest) if latest else None
        if latest:
            st.info(f"Используется файл из data/raw: {latest.name}")
        else:
            st.warning("Положите файл в data/raw или загрузите его выше.")
    st.markdown("---")
    st.header("Параметры анализа")
    peer_min_group_size = st.number_input(
        "Мин. число организаций-аналогов",
        min_value=3,
        max_value=100,
        value=5,
        step=1,
        help=(
            "Минимальный размер группы аналогов для сравнения в разделах «Форма обеспечения функционирования внутреннего контроля» "
            "и «Распределение 3D». При нехватке организаций применяется откат к более грубой группировке."
        ),
    )

    if st.button("Обновить данные", help="Очистить кэш загрузки отчёта и перечитать файл"):
        st.cache_data.clear()
        st.rerun()

if not input_path:
    st.info("Добро пожаловать в систему мониторинга организации и осуществления внутреннего контроля!")
    st.markdown("""
    ### 📊 Система анализа внутреннего контроля
    
    Для начала работы загрузите файл отчёта мониторинга через боковую панель слева
    
    #### Поддерживаемые форматы:
    - `.XLS` (старый формат Excel)
    - `.XLSX` (новый формат Excel)
    - `.HTML` / `.HTM` (экспорт из веб-форм)
    
    ---
    
    💡 **Совет:** После загрузки файла используйте фильтры в боковой панели для детального анализа по федеральным округам, субъектам Российской Федерации и типам организаций.
    """)
    st.stop()

try:
    df, scoring_config, resolution_report = load_and_score(
        input_path, int(year), min_group_size=int(peer_min_group_size)
    )
except Exception as e:
    st.error(f"Не удалось загрузить и обработать отчет: {e}")
    st.stop()

with st.sidebar:
    if "form_vs_peer" in df.columns:
        st.caption("Расчёт: сравнение с организациями аналогичной нагрузки ✓")
    else:
        st.caption("Расчёт: сравнение с организациями аналогичной нагрузки ✗")

    st.header("Фильтры")
    filtered = df.copy()
    for col, label in [
        ("federal_district", "Федеральный округ"),
        ("region", "Субъект Российской Федерации"),
        ("org_type_classified", "Тип организации"),
        ("risk_group", "Группа риска (расчётный показатель)"),
    ]:
        if col in filtered.columns:
            options = sorted([x for x in filtered[col].dropna().unique()])
            selected = st.multiselect(label, options, format_func=display_label if col == "risk_group" else str)
            if selected:
                filtered = filtered[filtered[col].isin(selected)]

ov = overview_metrics(filtered)

st.subheader("Ключевые показатели")
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Количество организаций", fmt_num(len(filtered)))
with col2:
    metric_card("Доля заполненных отчётов", pct(value_from_overview(ov, "% заполненных отчетов")))
with col3:
    metric_card("Внутренний контроль организован и осуществляется", pct(value_from_overview(ov, "% организаций, где СВК организован")))
with col4:
    metric_card("Организация внутреннего контроля полностью формализована", pct(value_from_overview(ov, "% организаций с полным базовым комплектом СВК")))

col5, col6, col7, col8 = st.columns(4)
with col5:
    metric_card("Методика по выявлению и оценке рисков применяется", pct(value_from_overview(ov, "% организаций с полноценной методикой оценки рисков")))
with col6:
    metric_card("Регулярная оценка эффективности организации и осуществления внутреннего контроля", pct(value_from_overview(ov, "% организаций с регулярной оценкой эффективности СВК")))
with col7:
    metric_card("Количество выявленных нарушений, ед.", fmt_num(value_from_overview(ov, "Выявленные нарушения")))
with col8:
    metric_card("Доля нарушений, устранённых в срок", pct(value_from_overview(ov, "% нарушений, устраненных в срок")))

col9, col10, col11, col12 = st.columns(4)
with col9:
    avg_coverage = filtered["coverage_share_active"].mean()
    metric_card("Доля активных направлений контроля, включённых в контур внутреннего контроля", pct(avg_coverage * 100) if not pd.isna(avg_coverage) else "—")
with col10:
    metric_card("Организации с непокрытыми активными направлениями", fmt_num(int((filtered["uncovered_active_directions_count"] > 0).sum())))
with col11:
    metric_card("Форма обеспечения функционирования внутреннего контроля ниже расчётной рекомендации", fmt_num(int((filtered["form_gap"] < 0).sum())))
with col12:
    metric_card("Организации с двойным расчётным риском", fmt_num(int(filtered["risk_group"].str.startswith("D.").sum())))


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Общая информация",
    "Зрелость ВК",
    "Направления контроля",
    "Соразмерность формы ВК",
    "Группы риска",
    "Качество отчёта",
    "Проверка колонок",
    "Распределение 3D",
])

with tab1:
    st.markdown("### Общая организация и осуществление внутреннего контроля")
    st.dataframe(with_display_labels(ov, ["metric"]), use_container_width=True)

    st.markdown("### Количественные показатели и работа с нарушениями")
    viol_remediation = violations_and_remediation(filtered)
    if not viol_remediation.empty:
        viol_plot = add_wrapped_display_column(viol_remediation, "metric", width=30)
        fig = px.bar(
            viol_plot,
            x="metric_display",
            y="value",
            text="value",
            title=chart_title("Количественные показатели и работа с нарушениями"),
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(viol_remediation, ["metric"]), use_container_width=True)

    st.markdown("### Управленческие меры (планы и решения)")
    mgmt_actions = management_actions(filtered)
    if not mgmt_actions.empty:
        mgmt_plot = add_wrapped_display_column(mgmt_actions, "metric", width=30)
        fig = px.bar(
            mgmt_plot,
            x="metric_display",
            y="value",
            text="value",
            title=chart_title("Планы и решения по устранению нарушений"),
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(mgmt_actions, ["metric"]), use_container_width=True)

    st.markdown("### Меры юридической ответственности")
    legal_measures = legal_responsibility_measures(filtered)
    if not legal_measures.empty:
        # Подготовка данных для сгруппированного графика
        chart_data = []
        
        # Дисциплинарная ответственность
        disciplinary_total = legal_measures.iloc[0]["value"]
        disciplinary_cancelled = legal_measures.iloc[1]["cancelled_returned_refused"]
        chart_data.append({"Мера": "Дисциплинарная\nответственность", "Тип": "Принято", "Количество": disciplinary_total})
        chart_data.append({"Мера": "Дисциплинарная\nответственность", "Тип": "Отменено", "Количество": disciplinary_cancelled})
        
        # Правоохранительные органы
        law_enforcement_total = legal_measures.iloc[2]["value"]
        law_enforcement_returned = legal_measures.iloc[3]["cancelled_returned_refused"]
        chart_data.append({"Мера": "Материалы в\nправоохранительные\nорганы", "Тип": "Направлено", "Количество": law_enforcement_total})
        chart_data.append({"Мера": "Материалы в\nправоохранительные\nорганы", "Тип": "Возвращено/отказ", "Количество": law_enforcement_returned})
        
        # Суд
        court_total = legal_measures.iloc[4]["value"]
        court_refused = legal_measures.iloc[5]["cancelled_returned_refused"]
        chart_data.append({"Мера": "Материалы в суд", "Тип": "Направлено", "Количество": court_total})
        chart_data.append({"Мера": "Материалы в суд", "Тип": "Отказано", "Количество": court_refused})
        
        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df, 
            x="Мера", 
            y="Количество", 
            color="Тип",
            text="Количество",
            title=chart_title("Меры юридической ответственности: принято решений и результаты"),
            barmode="group"
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # Показываем таблицу со всеми данными включая отмены/отказы и процент отмен
        st.caption("Детализация с процентами отмен/отказов:")
        display_df = legal_measures.copy()
        display_df = with_display_labels(display_df, ["metric"])
        # Форматируем процент отмен
        display_df["cancellation_rate_pct"] = display_df["cancellation_rate_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
        st.dataframe(display_df, use_container_width=True)

    st.markdown("### Нормированные показатели (расчётный показатель)")
    st.dataframe(with_display_labels(normalized_activity_metrics(filtered), ["metric"]), use_container_width=True)

with tab2:
    st.markdown("### Общая организация и осуществление внутреннего контроля")
    elems = svk_elements_summary(filtered)
    if not elems.empty:
        elems_plot = add_wrapped_display_column(elems, "element", width=34)
        fig = px.bar(
            elems_plot,
            x="element_display",
            y="yes_orgs",
            text="yes_orgs",
            title=chart_title("Наличие элементов организации и осуществления внутреннего контроля"),
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(elems, ["element", "metric"]), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Фактическая форма обеспечения функционирования внутреннего контроля")
        form_level = svk_form_level_summary(filtered)
        form_level_activity = svk_form_level_activity_summary(filtered)
        if not form_level.empty:
            form_level_plot = add_wrapped_display_column(form_level, "svk_form_name", "form_name_display", width=32)
            fig = px.bar(form_level_plot, x="form_name_display", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)

            if not form_level_activity.empty:
                display_act = with_display_labels(form_level_activity, ["svk_form_name"]).copy()

                # Форматируем процентные столбцы
                for pct_col in ["cancelled_decisions_rate_pct", "returned_refused_rate_pct", "refused_satisfaction_rate_pct"]:
                    if pct_col in display_act.columns:
                        display_act[pct_col] = display_act[pct_col].apply(
                            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                        )
                if "violations_per_org" in display_act.columns:
                    display_act["violations_per_org"] = display_act["violations_per_org"].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )
                if "percent_filled" in display_act.columns:
                    display_act["percent_filled"] = display_act["percent_filled"].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                    )

                rename_map = {
                    "orgs": "Организаций",
                    "percent_filled": "% от заполн.",
                    "violations_total": "Нарушений",
                    "violations_per_org": "Нарушений на орг.",
                    "disciplinary_decisions": "Дисципл. взыскания",
                    "cancelled_decisions": "из них отменены",
                    "cancelled_decisions_rate_pct": "% отмен (дисципл.)",
                    "materials_law_enforcement": "В правоохр. органы",
                    "returned_refused": "из них отказы (правоохр.)",
                    "returned_refused_rate_pct": "% отказов (правоохр.)",
                    "materials_court": "В суд",
                    "refused_satisfaction": "из них отказы (суд)",
                    "refused_satisfaction_rate_pct": "% отказов (суд)",
                }
                display_act = display_act.rename(columns={k: v for k, v in rename_map.items() if k in display_act.columns})

                st.caption("Нарушения и меры реагирования по форме обеспечения функционирования внутреннего контроля:")
                st.dataframe(display_act, use_container_width=True)
            else:
                st.dataframe(with_display_labels(form_level, ["svk_form_name"]), use_container_width=True)
    with right:
        st.markdown("### Способ обеспечения функционирования внутреннего контроля")
        form_flags = svk_form_flags_summary(filtered, scoring_config)
        if not form_flags.empty:
            form_flags_plot = add_wrapped_display_column(form_flags, "form", width=32)
            fig = px.bar(form_flags_plot, x="form_display", y="yes_orgs", text="yes_orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(with_display_labels(form_flags, ["form"]), use_container_width=True)

    st.markdown("### Риск-ориентированный подход")
    risk_m = risk_methodology_summary(filtered)
    if not risk_m.empty:
        risk_plot = add_wrapped_display_column(risk_m, "category", width=34)
        fig = px.bar(
            risk_plot,
            x="category_display",
            y="orgs",
            text="orgs",
            title=chart_title("Методика по выявлению и оценке рисков"),
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(risk_m, ["category"]), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Оценка эффективности организации и осуществления внутреннего контроля")
        eff = effectiveness_review_summary(filtered)
        if not eff.empty:
            eff_plot = add_wrapped_display_column(eff, "category", width=34)
            fig = px.bar(eff_plot, x="category_display", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(with_display_labels(eff, ["category"]), use_container_width=True)
    with col_b:
        st.markdown("### Совершенствование организации и осуществления внутреннего контроля")
        imp = improvement_actions_summary(filtered)
        if not imp.empty:
            imp_plot = add_wrapped_display_column(imp, "category", width=34)
            fig = px.bar(imp_plot, x="category_display", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(with_display_labels(imp, ["category"]), use_container_width=True)

with tab3:
    st.markdown("### Направления контроля")
    dsum = direction_summary(filtered, scoring_config)
    if not dsum.empty:
        chart_df = dsum.melt(
            id_vars=["direction"],
            value_vars=["covered_orgs", "uncovered_orgs"],
            var_name="status",
            value_name="orgs",
        )
        chart_df["status"] = chart_df["status"].map(lambda value: wrap_display_text(value, width=28))
        chart_df["direction_display"] = chart_df["direction"].map(lambda value: wrap_display_text(value, width=28))
        fig = px.bar(chart_df, x="direction_display", y="orgs", color="status", text="orgs", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(dsum, ["direction"]), use_container_width=True)

with tab4:
    st.markdown("### Сравнение с организациями аналогичной нагрузки (расчётный показатель)")
    st.caption(
        f"Сравнение с медианой формы обеспечения функционирования внутреннего контроля среди организаций с похожим профилем нагрузки "
        f"(мин. группа аналогов: {peer_min_group_size} орг.; при нехватке — более грубая группировка). "
        "Отклонение является расчётным аналитическим показателем, а не полем утверждённой формы мониторинга."
    )
    if "form_vs_peer" in filtered.columns:
        peer_plot = filtered.copy()
        peer_plot["form_vs_peer"] = peer_plot["form_vs_peer"].round().astype(int)
        form_peer_counts = (
            peer_plot.groupby("form_vs_peer", dropna=False)
            .size()
            .reset_index(name="orgs")
            .sort_values("form_vs_peer")
        )
        fig = px.bar(
            form_peer_counts,
            x="form_vs_peer",
            y="orgs",
            text="orgs",
            title=chart_title("Отклонение фактической формы обеспечения функционирования внутреннего контроля от медианы организаций-аналогов"),
        )
        fig.update_layout(
            xaxis_title=axis_title("Фактическая форма обеспечения функционирования внутреннего контроля − медиана организаций-аналогов (уровни)"),
            yaxis_title="Количество организаций",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "peer_form_median" in peer_plot.columns:
            peer_plot["peer_form_median"] = peer_plot["peer_form_median"].round().astype(int)
            peer_plot = add_wrapped_display_column(peer_plot, "risk_group", width=34)
            fig = px.scatter(
                peer_plot,
                x="peer_form_median",
                y="svk_form_level",
                color="risk_group_display",
                size="peer_group_size" if "peer_group_size" in peer_plot.columns else None,
                hover_name="org_name",
                title=chart_title("Фактическая форма обеспечения функционирования внутреннего контроля и медиана организаций-аналогов"),
            )
            fig.update_layout(
                xaxis_title=axis_title("Медиана формы обеспечения функционирования внутреннего контроля у организаций-аналогов"),
                yaxis_title=axis_title("Фактическая форма обеспечения функционирования внутреннего контроля"),
                legend_title=legend_title("Группа риска (расчётный показатель)"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Расчётные показатели сравнения с организациями аналогичной нагрузки не найдены. "
            "Перезапустите приложение: `streamlit run app.py`."
        )

    st.markdown("### Отклонение фактической формы обеспечения функционирования внутреннего контроля от расчётной рекомендации")
    gap_counts = filtered.groupby("form_gap", dropna=False).size().reset_index(name="orgs").sort_values("form_gap")
    fig = px.bar(
        gap_counts,
        x="form_gap",
        y="orgs",
        text="orgs",
        title=chart_title("Отклонение фактической формы обеспечения функционирования внутреннего контроля от расчётной рекомендации"),
    )
    fig.update_layout(
        xaxis_title=axis_title("Фактическая форма обеспечения функционирования внутреннего контроля − расчётная рекомендация"),
        yaxis_title="Количество организаций",
    )
    st.plotly_chart(fig, use_container_width=True)

    scatter_cols = ["org_name", "svk_form_level", "recommended_form_level", "max_direction_load_level", "coverage_share_active", "risk_group"]
    if all(c in filtered.columns for c in scatter_cols):
        filtered_scatter = add_wrapped_display_column(filtered, "risk_group", width=34)
        fig = px.scatter(
            filtered_scatter,
            x="recommended_form_level",
            y="svk_form_level",
            color="risk_group_display",
            size="max_direction_load_level",
            hover_name="org_name",
            title=chart_title("Фактическая форма обеспечения функционирования внутреннего контроля и расчётная рекомендация"),
        )
        fig.update_layout(
            xaxis_title=axis_title("Расчётная рекомендация по форме обеспечения функционирования внутреннего контроля"),
            yaxis_title=axis_title("Фактическая форма обеспечения функционирования внутреннего контроля"),
            legend_title=legend_title("Группа риска (расчётный показатель)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Разрез по федеральным округам")
    st.dataframe(with_display_labels(by_dimension_summary(filtered, "federal_district"), ["risk_group"]), use_container_width=True)

with tab5:
    st.markdown("### Организации для управленческого внимания")
    top = top_risk_organizations(filtered, limit=100)
    top_display = with_display_labels(top, ["risk_group", "svk_form_name", "recommended_form_name"])
    st.dataframe(top_display, use_container_width=True)
    st.download_button(
        "Скачать список организаций для управленческого внимания CSV",
        data=top.to_csv(index=False, encoding="utf-8-sig"),
        file_name="top_risk_organizations.csv",
        mime="text/csv",
    )

with tab6:
    st.markdown("### Организации с противоречиями в данных")
    st.caption("Логические несоответствия, требующие проверки и исправления")
    contradictions = contradictions_table(filtered)
    if not contradictions.empty:
        contradictions_display = with_display_labels(contradictions, ["contradictions", "issues", "risk_group", "svk_form_name"])
        st.dataframe(contradictions_display, use_container_width=True)
        st.download_button(
            "Скачать список с противоречиями CSV",
            data=contradictions.to_csv(index=False, encoding="utf-8-sig"),
            file_name="contradictions.csv",
            mime="text/csv",
        )
        st.info(f"Найдено организаций с противоречиями: {len(contradictions)}")
    else:
        st.success("Противоречий не обнаружено")
    
    st.markdown("---")
    
    st.markdown("### Организации с подозрениями на ошибки")
    st.caption("Аномальные значения и подозрительные паттерны данных")
    anomalies = anomalies_table(filtered)
    if not anomalies.empty:
        anomalies_display = with_display_labels(anomalies, ["issues", "risk_group", "svk_form_name", "recommended_form_name"])
        st.dataframe(anomalies_display, use_container_width=True)
        st.download_button(
            "Скачать список с аномалиями CSV",
            data=anomalies.to_csv(index=False, encoding="utf-8-sig"),
            file_name="anomalies.csv",
            mime="text/csv",
        )
        st.info(f"Найдено организаций с аномалиями: {len(anomalies)}")
    else:
        st.success("Аномалий не обнаружено")
    
    st.markdown("---")
    
    st.markdown("### Статус отчётности")
    status = report_status_summary(filtered)
    if not status.empty:
        status_plot = add_wrapped_display_column(status, "status", width=30)
        fig = px.bar(status_plot, x="status_display", y="orgs", text="orgs", title=chart_title("Распределение статусов отчёта"))
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(with_display_labels(status, ["status"]), use_container_width=True)

with tab7:
    st.markdown("### Проверка колонок отчёта")
    st.caption("Техническая проверка сопоставления колонок загруженного отчёта с ожидаемой структурой данных.")
    st.dataframe(with_display_text_columns(resolution_report), use_container_width=True)
    missing = resolution_report[~resolution_report["found"]]
    if len(missing):
        st.warning(f"Не найдено колонок: {len(missing)}")
    else:
        st.success("Все ключевые колонки найдены.")

with tab8:
    pa = proportionality_anomalies(filtered, scoring_config)
    if pa.empty or len(pa) < 5:
        st.info("Недостаточно заполненных отчётов в текущей выборке для анализа.")
    else:
        st.markdown("### Распределение организаций по трём осям")
        st.caption(
            "Общий вид без обработки: исходные значения суммы кассовых поступлений, "
            "среднесписочной численности и количества фактов ФХЖ; цвет — форма обеспечения функционирования внутреннего контроля "
            "(вид организации внутреннего контроля). Из-за сильного разброса большинство "
            "точек концентрируется у начала координат — это ожидаемо для «сырых» данных."
        )
        st.caption("Пояснение: ВК — внутренний контроль; «Форма функционирования ВК» — форма обеспечения функционирования внутреннего контроля.")
        hide_extremes = st.checkbox(
            "Скрывать экстремальные значения",
            value=True,
            help=(
                "Организации, превышающие порог хотя бы по одной из осей, исключаются "
                "из 3D-графиков (чтобы не растягивать шкалы) и выносятся в таблицу на проверку. "
                "Порог настраивается отдельно по каждой оси."
            ),
        )
        sld_cash, sld_staff, sld_fkhz = st.columns(3)
        with sld_cash:
            pct_cash = st.slider(
                "Порог: кассовые поступления, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not hide_extremes,
            )
        with sld_staff:
            pct_staff = st.slider(
                "Порог: численность, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not hide_extremes,
            )
        with sld_fkhz:
            pct_fkhz = st.slider(
                "Порог: факты ФХЖ, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not hide_extremes,
            )
        if hide_extremes:
            kept, extreme = split_extreme_values(
                pa,
                {
                    "cash_receipts": pct_cash / 100.0,
                    "staff_avg": pct_staff / 100.0,
                    "fkhz_count": pct_fkhz / 100.0,
                },
            )
        else:
            kept, extreme = pa, pa.iloc[0:0]

        log_cash_col, log_staff_col, log_fkhz_col = st.columns(3)
        with log_cash_col:
            log_cash = st.checkbox("lg по оси «Кассовые поступления»", value=False)
        with log_staff_col:
            log_staff = st.checkbox("lg по оси «Численность»", value=False)
        with log_fkhz_col:
            log_fkhz = st.checkbox("lg по оси «Факты ФХЖ»", value=False)
        st.caption(
            "3D-сцены Plotly не поддерживают лог-оси напрямую: при включении значения по оси "
            "пересчитываются в десятичный логарифм lg(x). Нулевые значения на лог-шкале не отображаются."
        )

        plot_raw = kept.copy()
        plot_raw["x_cash"] = (
            np.log10(plot_raw["cash_receipts"].where(plot_raw["cash_receipts"] > 0))
            if log_cash else plot_raw["cash_receipts"]
        )
        plot_raw["y_staff"] = (
            np.log10(plot_raw["staff_avg"].where(plot_raw["staff_avg"] > 0))
            if log_staff else plot_raw["staff_avg"]
        )
        plot_raw["z_fkhz"] = (
            np.log10(plot_raw["fkhz_count"].where(plot_raw["fkhz_count"] > 0))
            if log_fkhz else plot_raw["fkhz_count"]
        )
        plot_raw = add_wrapped_display_column(plot_raw, "svk_form_name", "form_name_display", width=30)
        x_title = "Кассовые поступления, lg(руб.)" if log_cash else "Кассовые поступления, руб."
        y_title = "Среднесписочная численность, lg(чел.)" if log_staff else "Среднесписочная численность"
        z_title = "Количество фактов ФХЖ, lg(ед.)" if log_fkhz else "Количество фактов ФХЖ"
        raw3d = px.scatter_3d(
            plot_raw,
            x="x_cash",
            y="y_staff",
            z="z_fkhz",
            color="form_name_display",
            hover_name="org_name",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={
                "x_cash": False,
                "y_staff": False,
                "z_fkhz": False,
                "cash_receipts": ":,.0f",
                "staff_avg": ":,.0f",
                "fkhz_count": ":,.0f",
            },
            title=chart_title("Кассовые поступления × численность × факты ФХЖ (по форме обеспечения функционирования внутреннего контроля)"),
        )
        raw3d.update_layout(
            legend_title="Форма функционирования ВК",
            scene=dict(
                xaxis=dict(title=x_title),
                yaxis=dict(title=y_title),
                zaxis=dict(title=z_title),
            ),
            height=700,
        )
        st.plotly_chart(raw3d, use_container_width=True)

        if hide_extremes and not extreme.empty:
            st.markdown("#### Экстремальные значения — вынесены на проверку")
            st.caption(
                "Организации, превышающие порог хотя бы по одной из трёх осей. Исключены из "
                "3D-графиков, чтобы не растягивать шкалы; рекомендуется проверить исходные данные."
            )
            ext_cols = [
                c
                for c in [
                    "org_name", "org_type_classified", "federal_district", "region",
                    "cash_receipts", "staff_avg", "fkhz_count",
                    "svk_form_name", "extreme_reason",
                ]
                if c in extreme.columns
            ]
            st.dataframe(
                with_display_labels(extreme[ext_cols], ["svk_form_name", "extreme_reason"]),
                use_container_width=True,
            )
            st.download_button(
                "Скачать экстремальные значения CSV",
                data=extreme[ext_cols].to_csv(index=False, encoding="utf-8-sig"),
                file_name="extreme_values.csv",
                mime="text/csv",
            )
            st.info(
                f"Вынесено организаций: {len(extreme)} из {len(pa)} "
                f"(пороги: поступления {pct_cash:.1f}%, численность {pct_staff:.1f}%, "
                f"ФХЖ {pct_fkhz:.1f}%)."
            )

        st.markdown("---")
        st.markdown("### Профиль масштаба: концентрация и форма обеспечения функционирования внутреннего контроля")
        st.caption(
            "Ось масштаба — направление роста по трём показателям ФХД в ln(1+x)-пространстве "
            "(первая главная компонента). Отрицательные значения — мельче типичного по оси, "
            "положительные — крупнее. Графики показывают, где сосредоточены организации и как "
            "меняется медиана формы обеспечения функционирования внутреннего контроля по масштабу. Карточки «типичных» показателей и вертикальная "
            "линия на графике — бин с максимальной концентрацией организаций. "
            "Организации без положительных значений по всем трём осям исключаются из расчёта оси и бинов."
        )
        st.caption("Пояснение: ВК — внутренний контроль; «Медиана формы ВК» — медиана формы обеспечения функционирования внутреннего контроля.")

        trim_profile = st.checkbox(
            "Отсекать выбросы профиля",
            value=True,
            key="sp_trim_outliers",
            help=(
                "Организации ниже нижнего или выше верхнего перцентильного порога хотя бы по одной "
                "из трёх осей исключаются из расчёта профиля масштаба и выносятся в таблицу на проверку."
            ),
        )
        sp_low_cash, sp_low_staff, sp_low_fkhz = st.columns(3)
        with sp_low_cash:
            sp_pct_low_cash = st.slider(
                "Нижний порог: кассовые поступления, перцентиль",
                min_value=0.1, max_value=10.0, value=0.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_low_cash",
            )
        with sp_low_staff:
            sp_pct_low_staff = st.slider(
                "Нижний порог: численность, перцентиль",
                min_value=0.1, max_value=10.0, value=0.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_low_staff",
            )
        with sp_low_fkhz:
            sp_pct_low_fkhz = st.slider(
                "Нижний порог: факты ФХЖ, перцентиль",
                min_value=0.1, max_value=10.0, value=0.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_low_fkhz",
            )
        sp_high_cash, sp_high_staff, sp_high_fkhz = st.columns(3)
        with sp_high_cash:
            sp_pct_high_cash = st.slider(
                "Верхний порог: кассовые поступления, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_high_cash",
            )
        with sp_high_staff:
            sp_pct_high_staff = st.slider(
                "Верхний порог: численность, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_high_staff",
            )
        with sp_high_fkhz:
            sp_pct_high_fkhz = st.slider(
                "Верхний порог: факты ФХЖ, перцентиль",
                min_value=90.0, max_value=99.9, value=99.5, step=0.1,
                disabled=not trim_profile,
                key="sp_pct_high_fkhz",
            )

        if trim_profile:
            sp_kept, sp_trimmed = split_profile_outliers(
                filtered,
                low_percentiles={
                    "cash_receipts": sp_pct_low_cash / 100.0,
                    "staff_avg": sp_pct_low_staff / 100.0,
                    "fkhz_count": sp_pct_low_fkhz / 100.0,
                },
                high_percentiles={
                    "cash_receipts": sp_pct_high_cash / 100.0,
                    "staff_avg": sp_pct_high_staff / 100.0,
                    "fkhz_count": sp_pct_high_fkhz / 100.0,
                },
            )
        else:
            sp_kept, sp_trimmed = filtered, filtered.iloc[0:0]

        sp_n_bins_default = int(scoring_config.get("scale_profile", {}).get("n_bins", 12))
        sp_n_bins = st.slider(
            "Число бинов по оси масштаба",
            min_value=6,
            max_value=24,
            value=min(max(sp_n_bins_default, 6), 24),
            step=1,
            key="sp_n_bins",
            help="Больше бинов — уже интервалы (детальнее); меньше — шире (обобщённее).",
        )
        sp = scale_axis_profile(sp_kept, scoring_config, n_bins=sp_n_bins)
        sp_bins_count = sp.get("bins_count", sp.get("bins", pd.DataFrame()))
        sp_bins_share = sp.get("bins_share", sp.get("bins", pd.DataFrame()))
        sp_bin_orgs = sp.get("bin_orgs", pd.DataFrame())
        sp_orgs = sp["orgs"]
        sp_active = (
            sp_orgs.loc[sp_orgs["active"]]
            if "active" in sp_orgs.columns
            else sp_orgs.dropna(subset=["signed_scale"])
        )
        _FORM_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
        _FORM_COLORSCALE = [[i / 4, c] for i, c in enumerate(_FORM_COLORS)]

        if sp_bins_count.empty and sp_bins_share.empty:
            st.info(
                "Недостаточно данных для профиля масштаба "
                "(нужно не менее 20 заполненных отчётов с положительными значениями по всем трём осям ФХД)."
            )
        else:
            inactive_n = int((~sp_orgs["active"]).sum()) if "active" in sp_orgs.columns else 0
            if inactive_n:
                st.caption(
                    f"Исключено из расчёта оси масштаба и бинов: {inactive_n} орг. "
                    "(нет положительных значений по всем трём осям)."
                )

            scores_std = float(sp_active["signed_scale"].std(ddof=0)) if not sp_active.empty else 1.0
            if scores_std <= 0:
                scores_std = 1.0

            if not sp_bins_count.empty:
                peak_row = sp_bins_count.loc[sp_bins_count["n"].idxmax()]
                center_cash = float(peak_row["typical_cash"])
                center_staff = float(peak_row["typical_staff"])
                center_fkhz = float(peak_row["typical_fkhz"])
            elif not sp_active.empty:
                center_cash = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["cash_receipts"], errors="coerce").fillna(0)).mean()
                )
                center_staff = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["staff_avg"], errors="coerce").fillna(0)).mean()
                )
                center_fkhz = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["fkhz_count"], errors="coerce").fillna(0)).mean()
                )
            else:
                center_cash = center_staff = center_fkhz = None

            if center_cash is not None:
                k1, k2, k3 = st.columns(3)
                with k1:
                    metric_card("Типичные поступления", f"≈ {center_cash / 1e6:,.0f} млн руб.")
                with k2:
                    metric_card("Типичная численность", f"≈ {center_staff:,.0f} чел.")
                with k3:
                    metric_card("Типичные факты ФХЖ", f"≈ {center_fkhz:,.0f} ед.")
                if not sp_bins_count.empty:
                    st.caption(
                        f"По бину максимальной концентрации: n={int(peak_row['n'])}, "
                        f"медиана формы обеспечения функционирования внутреннего контроля {int(peak_row['form_median'])}."
                    )

            def _sigma_label(mid: float) -> str:
                if abs(mid) < 1e-9:
                    return "0"
                sigma = mid / scores_std
                if abs(sigma) < 0.05:
                    return "0"
                rounded = round(sigma, 1)
                if rounded == 0:
                    return "0"
                prefix = "+" if rounded > 0 else "−"
                return f"{prefix}{abs(rounded)}σ"

            def _typical_label(row: pd.Series) -> str:
                return (
                    f"≈ {row['typical_cash'] / 1e6:,.0f} млн руб / "
                    f"{row['typical_staff']:,.0f} чел / {row['typical_fkhz']:,.0f} ед."
                )

            def _scale_x_axis(bins_df: pd.DataFrame) -> tuple[list[int], dict, int]:
                x_labels = [
                    f"{_sigma_label(row['signed_scale_mid'])}<br>{_typical_label(row)}"
                    for _, row in bins_df.iterrows()
                ]
                x_pos = list(range(len(bins_df)))
                x_axis = dict(tickmode="array", tickvals=x_pos, ticktext=x_labels)
                peak_idx = int(bins_df["n"].to_numpy().argmax())
                return x_pos, x_axis, peak_idx

            def _scale_range_label(left: float, right: float) -> str:
                return f"{left:.2f} … {right:.2f}"

            def _cash_range_label(lo: float, hi: float) -> str:
                return f"{lo / 1e6:,.1f}–{hi / 1e6:,.1f} млн руб"

            def _bin_hover_text(row: pd.Series, low: bool) -> str:
                lines = [
                    f"Медиана формы обеспечения функционирования внутреннего контроля: {row['form_median']:.0f} | n={int(row['n'])}",
                    f"Масштаб: {_scale_range_label(row['signed_scale_left'], row['signed_scale_right'])}",
                    f"Поступления: {_cash_range_label(row['cash_min'], row['cash_max'])}",
                    f"Численность: {row['staff_min']:,.0f}–{row['staff_max']:,.0f} чел",
                    f"ФХЖ: {row['fkhz_min']:,.0f}–{row['fkhz_max']:,.0f} ед",
                ]
                if low:
                    lines.append("<b>n мало</b>")
                return "<br>".join(lines)

            scale_bins = sp_bins_count if not sp_bins_count.empty else sp_bins_share
            if not scale_bins.empty:
                x_pos_scale, x_axis_scale, peak_idx_scale = _scale_x_axis(scale_bins)
                share_df = sp_bins_share if not sp_bins_share.empty else sp_bins_count

            if not sp_bins_count.empty:
                x_pos_count, x_axis_count, peak_idx_count = x_pos_scale, x_axis_scale, peak_idx_scale
                bar_opacity = [0.5 if low else 1.0 for low in sp_bins_count["low_n"]]
                bar_hover = [
                    _bin_hover_text(sp_bins_count.iloc[i], bool(sp_bins_count["low_n"].iloc[i]))
                    for i in range(len(sp_bins_count))
                ]

                fig_scale_n = go.Figure(
                    go.Bar(
                        x=x_pos_count,
                        y=sp_bins_count["n"],
                        marker=dict(
                            color=sp_bins_count["form_median"],
                            colorscale=_FORM_COLORSCALE,
                            cmin=0,
                            cmax=4,
                            opacity=bar_opacity,
                            colorbar=dict(
                                title="Медиана формы ВК",
                                tickvals=list(range(5)),
                                ticktext=[wrap_display_text(display_form_name(i), width=28) for i in range(5)],
                            ),
                        ),
                        text=[
                            f"{m:.0f} | n={n}"
                            for m, n in zip(sp_bins_count["form_median"], sp_bins_count["n"])
                        ],
                        textposition="outside",
                        hovertext=bar_hover,
                        hoverinfo="text",
                    )
                )
                fig_scale_n.add_vline(
                    x=peak_idx_count,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Максимум концентрации",
                    annotation_position="top",
                )
                fig_scale_n.update_layout(
                    title="Распределение организаций по масштабу",
                    xaxis=dict(title="Позиция по оси масштаба", **x_axis_count),
                    yaxis_title="Число организаций",
                    height=480,
                )
                st.plotly_chart(fig_scale_n, use_container_width=True)

            if not sp_bins_count.empty and not share_df.empty:
                fig_scale_share = go.Figure()
                for level in range(5):
                    share_pct = share_df[f"form_share_{level}"] * 100
                    count_in_bin = (share_df[f"form_share_{level}"] * share_df["n"]).round(0)
                    fig_scale_share.add_trace(
                        go.Bar(
                            name=f"{level} — {wrap_display_text(display_form_name(level), width=28)}",
                            x=x_pos_scale,
                            y=share_pct,
                            marker_color=_FORM_COLORS[level],
                            customdata=count_in_bin,
                            hovertemplate=(
                                f"Форма обеспечения функционирования внутреннего контроля {level}<br>"
                                "Доля: %{y:.1f}%<br>"
                                "Организаций: %{customdata:.0f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                fig_scale_share.add_vline(
                    x=peak_idx_scale,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Максимум концентрации",
                    annotation_position="top",
                )
                fig_scale_share.update_layout(
                    barmode="stack",
                    title=chart_title("Структура форм обеспечения функционирования внутреннего контроля по уровням масштаба"),
                    xaxis=dict(title="Позиция по оси масштаба", **x_axis_scale),
                    yaxis_title="Доля организаций, %",
                    yaxis=dict(range=[0, 100]),
                    legend_title="Форма функционирования ВК",
                    height=480,
                )
                st.plotly_chart(fig_scale_share, use_container_width=True)

            if not sp_bins_count.empty:
                st.markdown("#### Состав бинов: границы показателей")
                st.caption(
                    "Для каждого столбца гистограммы — диапазон позиции по оси масштаба "
                    "и min/max трёх показателей ФХД среди организаций бина."
                )
                bins_summary = sp_bins_count[
                    [
                        "bin_idx",
                        "signed_scale_left",
                        "signed_scale_right",
                        "cash_min",
                        "cash_max",
                        "staff_min",
                        "staff_max",
                        "fkhz_min",
                        "fkhz_max",
                        "n",
                        "form_median",
                    ]
                ].copy()
                bins_summary.insert(0, "Бин", bins_summary["bin_idx"] + 1)
                bins_summary["Масштаб"] = bins_summary.apply(
                    lambda r: _scale_range_label(r["signed_scale_left"], r["signed_scale_right"]),
                    axis=1,
                )
                bins_summary["Поступления, млн руб"] = bins_summary.apply(
                    lambda r: f"{r['cash_min'] / 1e6:,.1f} – {r['cash_max'] / 1e6:,.1f}",
                    axis=1,
                )
                bins_summary["Численность, чел"] = bins_summary.apply(
                    lambda r: f"{r['staff_min']:,.0f} – {r['staff_max']:,.0f}",
                    axis=1,
                )
                bins_summary["ФХЖ, ед"] = bins_summary.apply(
                    lambda r: f"{r['fkhz_min']:,.0f} – {r['fkhz_max']:,.0f}",
                    axis=1,
                )
                bins_summary = bins_summary.rename(columns={"n": "Организаций", "form_median": "Медиана формы"})
                st.dataframe(
                    bins_summary[
                        [
                            "Бин",
                            "Масштаб",
                            "Поступления, млн руб",
                            "Численность, чел",
                            "ФХЖ, ед",
                            "Организаций",
                            "Медиана формы",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                if not sp_bin_orgs.empty:
                    bin_options = {
                        int(row["bin_idx"]): (
                            f"Бин {int(row['bin_idx']) + 1}: "
                            f"{_sigma_label(row['signed_scale_mid'])} "
                            f"({int(row['n'])} орг.)"
                        )
                        for _, row in sp_bins_count.iterrows()
                    }
                    selected_bin_idx = st.selectbox(
                        "Организации в бине",
                        options=list(bin_options.keys()),
                        format_func=lambda idx: bin_options[idx],
                        key="sp_bin_orgs_select",
                    )
                    bin_detail_cols = [
                        c
                        for c in [
                            "org_name",
                            "cash_receipts",
                            "staff_avg",
                            "fkhz_count",
                            "svk_form_name",
                            "signed_scale",
                        ]
                        if c in sp_bin_orgs.columns
                    ]
                    bin_detail = sp_bin_orgs.loc[
                        sp_bin_orgs["bin_idx"] == selected_bin_idx, bin_detail_cols
                    ].sort_values("signed_scale")
                    st.dataframe(
                        with_display_labels(bin_detail, ["svk_form_name"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Скачать организации бина CSV",
                        data=bin_detail.to_csv(index=False, encoding="utf-8-sig"),
                        file_name=f"scale_profile_bin_{selected_bin_idx + 1}.csv",
                        mime="text/csv",
                        key="sp_bin_orgs_csv",
                    )

                    # --- Агрегация по формам внутри выбранного бина ---
                    _bin_metric_cols = [
                        c for c in [
                            "violations_total",
                            "disciplinary_decisions",
                            "cancelled_decisions",
                            "materials_law_enforcement",
                            "returned_refused",
                            "materials_court",
                            "refused_satisfaction",
                        ]
                        if c in sp_bin_orgs.columns
                    ]
                    _bin_full = sp_bin_orgs.loc[sp_bin_orgs["bin_idx"] == selected_bin_idx].copy()
                    if _bin_metric_cols and "svk_form_name" in _bin_full.columns:
                        for _c in _bin_metric_cols:
                            _bin_full[_c] = pd.to_numeric(_bin_full[_c], errors="coerce").fillna(0)

                        _form_agg = (
                            _bin_full.groupby("svk_form_name", dropna=False)
                            .agg(
                                **{"Организаций": pd.NamedAgg("svk_form_name", "count")},
                                **{_c: pd.NamedAgg(_c, "sum") for _c in _bin_metric_cols},
                            )
                            .reset_index()
                            .sort_values("Организаций", ascending=False)
                        )

                        if "violations_total" in _form_agg.columns:
                            _form_agg["Нарушений на орг."] = (
                                _form_agg["violations_total"]
                                / _form_agg["Организаций"].replace(0, np.nan)
                            ).apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

                        for _num, _den, _col in [
                            ("cancelled_decisions", "disciplinary_decisions", "% отмен (дисципл.)"),
                            ("returned_refused", "materials_law_enforcement", "% отказов (правоохр.)"),
                            ("refused_satisfaction", "materials_court", "% отказов (суд)"),
                        ]:
                            if _num in _form_agg.columns and _den in _form_agg.columns:
                                _form_agg[_col] = (
                                    _form_agg[_num] / _form_agg[_den].replace(0, np.nan) * 100
                                ).apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

                        _bin_rename = {
                            "svk_form_name": "Форма ВК",
                            "violations_total": "Нарушений",
                            "disciplinary_decisions": "Дисципл. взыскания",
                            "cancelled_decisions": "из них отменены",
                            "materials_law_enforcement": "В правоохр. органы",
                            "returned_refused": "из них отказы (правоохр.)",
                            "materials_court": "В суд",
                            "refused_satisfaction": "из них отказы (суд)",
                        }
                        _form_agg = _form_agg.rename(
                            columns={k: v for k, v in _bin_rename.items() if k in _form_agg.columns}
                        )

                        st.caption("Формы ВК в бине и их результативность:")
                        st.dataframe(_form_agg, use_container_width=True, hide_index=True)

        if trim_profile and not sp_trimmed.empty:
            st.markdown("#### Отсечённые организации — вынесены на проверку")
            st.caption(
                "Организации ниже нижнего или выше верхнего перцентильного порога хотя бы по одной "
                "из трёх осей. Исключены из расчёта профиля масштаба; рекомендуется проверить исходные данные."
            )
            sp_trim_cols = [
                c
                for c in [
                    "org_name", "org_type_classified", "federal_district", "region",
                    "cash_receipts", "staff_avg", "fkhz_count",
                    "svk_form_name", "profile_trim_reason",
                ]
                if c in sp_trimmed.columns
            ]
            st.dataframe(
                with_display_labels(sp_trimmed[sp_trim_cols], ["svk_form_name", "profile_trim_reason"]),
                use_container_width=True,
            )
            st.download_button(
                "Скачать отсечённые организации CSV",
                data=sp_trimmed[sp_trim_cols].to_csv(index=False, encoding="utf-8-sig"),
                file_name="scale_profile_trimmed.csv",
                mime="text/csv",
                key="sp_trimmed_csv",
            )
            st.info(
                f"Отсечено организаций: {len(sp_trimmed)} из {len(filtered)} "
                f"(нижние пороги: поступления {sp_pct_low_cash:.1f}%, численность {sp_pct_low_staff:.1f}%, "
                f"ФХЖ {sp_pct_low_fkhz:.1f}%; верхние: {sp_pct_high_cash:.1f}%, {sp_pct_high_staff:.1f}%, "
                f"{sp_pct_high_fkhz:.1f}%)."
            )

        st.markdown("---")

        st.markdown("### Сравнение с организациями-аналогами в облаке (расчётный показатель)")
        st.caption(
            f"Для каждой организации находятся {int(scoring_config.get('peer_benchmark', {}).get('min_group_size', 5))} "
            "ближайших соседей в пространстве ln(1+поступления), ln(1+численность), ln(1+ФХЖ). "
            "Сравнивается фактическая форма обеспечения функционирования внутреннего контроля с медианой формы у соседей. "
            "Перекос профиля показывает, по какой оси масштаб отличается от типичного у соседей; "
            "доверие к сравнению выше, когда соседи близко и их достаточно."
        )
        st.caption("Пояснение: ВК — внутренний контроль; «Форма функционирования ВК» — форма обеспечения функционирования внутреннего контроля.")

        cp_full = cloud_peer_benchmark(filtered, scoring_config)
        if "org_name" in kept.columns and "org_name" in cp_full.columns:
            cp = cp_full[cp_full["org_name"].isin(kept["org_name"])].copy()
        else:
            cp = cp_full.copy()
        min_k = int(scoring_config.get("peer_benchmark", {}).get("min_group_size", 5))
        if cp_full.empty or len(cp_full) < min_k + 1:
            st.info(
                f"Недостаточно организаций для сравнения с аналогами в облаке "
                f"(нужно не менее {min_k + 1}, сейчас {len(cp_full)})."
            )
        else:
            below_peer = cp[
                (cp["form_vs_peer_3d"] <= -1) & (cp["peer_3d_confidence"] >= 0.5)
            ]
            avg_conf = cp["peer_3d_confidence"].mean()

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Организаций в анализе", fmt_num(len(cp)))
            with c2:
                metric_card("Форма обеспечения функционирования внутреннего контроля ниже соседей (надёжно)", fmt_num(len(below_peer)))
            with c3:
                metric_card("Среднее доверие к сравнению", fmt_num(avg_conf, 2) if pd.notna(avg_conf) else "—")

            plot_cp = cp.copy()
            plot_cp["log_cash"] = np.log1p(plot_cp["cash_receipts"])
            plot_cp["log_staff"] = np.log1p(plot_cp["staff_avg"])
            plot_cp["log_fkhz"] = np.log1p(plot_cp["fkhz_count"])

            st.markdown("#### 3D-облако: отклонение формы обеспечения функционирования внутреннего контроля от соседей")
            fig3d = px.scatter_3d(
                plot_cp,
                x="log_cash",
                y="log_staff",
                z="log_fkhz",
                color="form_vs_peer_3d",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                opacity=0.8,
                hover_name="org_name",
                hover_data={
                    "log_cash": False,
                    "log_staff": False,
                    "log_fkhz": False,
                    "cash_receipts": ":,.0f",
                    "staff_avg": ":,.0f",
                    "fkhz_count": ":,.0f",
                    "peer_3d_form_median": True,
                    "form_vs_peer_3d": True,
                    "profile_skew_3d": ":.3f",
                    "profile_skew_3d_axis": True,
                    "peer_3d_confidence": ":.2f",
                },
                title=chart_title("Отклонение формы обеспечения функционирования внутреннего контроля от медианы ближайших соседей"),
            )
            fig3d.update_layout(
                coloraxis_colorbar_title="Форма функционирования ВК",
                scene=dict(
                    xaxis_title="ln(1+кассовые поступления)",
                    yaxis_title="ln(1+численность)",
                    zaxis_title="ln(1+факты ФХЖ)",
                ),
                height=700,
            )
            st.plotly_chart(fig3d, use_container_width=True)

            org_options = sorted(plot_cp["org_name"].dropna().astype(str).unique())
            selected_org = st.selectbox(
                "Подсветить соседей для организации",
                options=["—"] + org_options,
                index=0,
            )
            if selected_org != "—":
                neighbors = get_cloud_peer_neighbor_names(cp_full, selected_org)
                neighbor_set = set(neighbors)
                hi = plot_cp.copy()
                names = hi["org_name"].astype(str)
                hi["Группа"] = np.select(
                    [
                        names == selected_org,
                        names.isin(neighbor_set),
                    ],
                    ["Выбранная организация", "Соседи"],
                    default="Прочие",
                )
                fig_hi = px.scatter_3d(
                    hi,
                    x="log_cash",
                    y="log_staff",
                    z="log_fkhz",
                    color="Группа",
                    category_orders={
                        "Группа": ["Выбранная организация", "Соседи", "Прочие"],
                    },
                    color_discrete_map={
                        "Выбранная организация": "#f39c12",
                        "Соседи": "#3498db",
                        "Прочие": "#bdbdbd",
                    },
                    opacity=0.85,
                    hover_name="org_name",
                    title=f"Соседи в облаке: {selected_org}",
                )
                fig_hi.update_layout(
                    scene=dict(
                        xaxis_title="ln(1+кассовые поступления)",
                        yaxis_title="ln(1+численность)",
                        zaxis_title="ln(1+факты ФХЖ)",
                    ),
                    height=650,
                )
                st.plotly_chart(fig_hi, use_container_width=True)
                if neighbors:
                    st.caption(f"Соседей: {len(neighbors)} — {', '.join(neighbors[:5])}" + (" …" if len(neighbors) > 5 else ""))

            st.markdown("#### Фактическая форма обеспечения функционирования внутреннего контроля и медиана соседей")
            peer_scatter = cp.dropna(subset=["peer_3d_form_median", "svk_form_level"]).copy()
            if not peer_scatter.empty:
                fig_peer = px.scatter(
                    peer_scatter,
                    x="peer_3d_form_median",
                    y="svk_form_level",
                    color="form_vs_peer_3d",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    size="peer_3d_confidence",
                    size_max=18,
                    hover_name="org_name",
                    title=chart_title("Фактическая форма обеспечения функционирования внутреннего контроля и медиана ближайших соседей"),
                )
                fig_peer.update_layout(
                    xaxis_title=axis_title("Медиана формы обеспечения функционирования внутреннего контроля у соседей (0–4)"),
                    yaxis_title=axis_title("Фактическая форма обеспечения функционирования внутреннего контроля (0–4)"),
                )
                max_level = max(4, int(peer_scatter["svk_form_level"].max()), int(peer_scatter["peer_3d_form_median"].max()))
                fig_peer.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=max_level,
                    y1=max_level,
                    line=dict(color="gray", dash="dash"),
                    layer="below",
                )
                st.plotly_chart(fig_peer, use_container_width=True)

            st.markdown("#### Организации с формой обеспечения функционирования внутреннего контроля ниже соседей")
            priority_tab = cloud_peer_benchmark_table(filtered, scoring_config)
            if priority_tab.empty:
                st.success("В текущей выборке приоритетных отклонений не обнаружено.")
            else:
                priority_display = with_display_labels(priority_tab, ["svk_form_name", "recommended_form_name", "risk_group"])
                st.dataframe(priority_display, use_container_width=True)
                st.download_button(
                    "Скачать расчёт сравнения 3D CSV",
                    data=cp_full.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="cloud_peer_3d.csv",
                    mime="text/csv",
                )
                st.info(f"Найдено организаций с формой обеспечения функционирования внутреннего контроля ниже соседей (надёжное сравнение): {len(priority_tab)}")
