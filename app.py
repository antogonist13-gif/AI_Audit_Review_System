from __future__ import annotations

from pathlib import Path
import tempfile

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
    report_status_summary,
    risk_group_summary,
    risk_methodology_summary,
    svk_elements_summary,
    svk_form_flags_summary,
    svk_form_level_summary,
    top_risk_organizations,
    violations_and_remediation,
    violations_summary,
)

st.set_page_config(page_title="СВК Analytics", layout="wide")


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


def metric_card(label: str, value):
    st.metric(label, value)


st.title("СВК Analytics: мониторинг внутреннего контроля")
st.caption("Интерактивный дашборд по ежегодному сводному отчету")

with st.sidebar:
    st.header("Источник данных")
    year = st.number_input("Год анализа", min_value=2020, max_value=2100, value=2025, step=1)
    uploaded = st.file_uploader("Загрузите годовой отчет", type=["xls", "xlsx", "html", "htm"])

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
            "Минимальный размер группы аналогов для сравнения в разделах «Соразмерность формы» "
            "и «Распределение 3D». При нехватке организаций применяется откат к более грубой группировке."
        ),
    )

    if st.button("Обновить данные", help="Очистить кэш загрузки отчёта и перечитать файл"):
        st.cache_data.clear()
        st.rerun()

if not input_path:
    st.info("👋 Добро пожаловать в СВК Analytics!")
    st.markdown("""
    ### 📊 Система анализа внутреннего контроля
    
    Для начала работы загрузите файл отчета через боковую панель слева ⬅️
    
    #### Поддерживаемые форматы:
    - `.XLS` (старый формат Excel)
    - `.XLSX` (новый формат Excel)
    - `.HTML` / `.HTM` (экспорт из веб-форм)
    
    #### Что вы получите:
    - 📈 **12 ключевых показателей** эффективности СВК
    - 📊 **7 интерактивных вкладок** с детальной аналитикой
    - 🔍 **Автоматическое выявление** противоречий и аномалий в данных
    - 🎯 **Топ-100 организаций** в зоне риска
    - 📥 **Экспорт результатов** в CSV
    
    #### Направления анализа:
    1. Финансово-хозяйственная деятельность
    2. Закупки
    3. Имущество
    4. Проектная деятельность
    
    ---
    
    💡 **Совет:** После загрузки файла используйте фильтры в боковой панели для детального анализа по округам, регионам и типам организаций.
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
        st.caption("Расчёт: peer-метрики ✓")
    else:
        st.caption("Расчёт: peer-метрики ✗ — проверьте ветку `feature/peer-metrics-recommended-form`")

    st.header("Фильтры")
    filtered = df.copy()
    for col, label in [("federal_district", "Федеральный округ"), ("region", "Регион"), ("org_type_classified", "Тип организации"), ("risk_group", "Группа риска")]:
        if col in filtered.columns:
            options = sorted([x for x in filtered[col].dropna().unique()])
            selected = st.multiselect(label, options)
            if selected:
                filtered = filtered[filtered[col].isin(selected)]

ov = overview_metrics(filtered)

st.subheader("Ключевые показатели")
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Организаций", fmt_num(len(filtered)))
with col2:
    metric_card("Заполненность отчетов", pct(value_from_overview(ov, "% заполненных отчетов")))
with col3:
    metric_card("СВК организован", pct(value_from_overview(ov, "% организаций, где СВК организован")))
with col4:
    metric_card("Полный базовый комплект", pct(value_from_overview(ov, "% организаций с полным базовым комплектом СВК")))

col5, col6, col7, col8 = st.columns(4)
with col5:
    metric_card("Полноценная методика рисков", pct(value_from_overview(ov, "% организаций с полноценной методикой оценки рисков")))
with col6:
    metric_card("Регулярная оценка СВК", pct(value_from_overview(ov, "% организаций с регулярной оценкой эффективности СВК")))
with col7:
    metric_card("Нарушения", fmt_num(value_from_overview(ov, "Выявленные нарушения")))
with col8:
    metric_card("Устранено в срок", pct(value_from_overview(ov, "% нарушений, устраненных в срок")))

col9, col10, col11, col12 = st.columns(4)
with col9:
    avg_coverage = filtered["coverage_share_active"].mean()
    metric_card("Покрытие активных направлений", pct(avg_coverage * 100) if not pd.isna(avg_coverage) else "—")
with col10:
    metric_card("Непокрытые направления", fmt_num(int((filtered["uncovered_active_directions_count"] > 0).sum())))
with col11:
    metric_card("Слабая форма СВК", fmt_num(int((filtered["form_gap"] < 0).sum())))
with col12:
    metric_card("Двойной риск", fmt_num(int(filtered["risk_group"].str.startswith("D.").sum())))


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Общая статистика",
    "Зрелость СВК",
    "Направления",
    "Соразмерность формы",
    "Организации в зоне риска",
    "Качество данных",
    "Настройки колонок",
    "Распределение 3D",
])

with tab1:
    st.markdown("### Общие показатели")
    st.dataframe(ov, use_container_width=True)

    st.markdown("### Нарушения и их устранение")
    viol_remediation = violations_and_remediation(filtered)
    if not viol_remediation.empty:
        fig = px.bar(viol_remediation, x="metric", y="value", text="value", title="Нарушения и их устранение")
        fig.update_layout(xaxis_title="", yaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(viol_remediation, use_container_width=True)

    st.markdown("### Управленческие меры (планы и решения)")
    mgmt_actions = management_actions(filtered)
    if not mgmt_actions.empty:
        fig = px.bar(mgmt_actions, x="metric", y="value", text="value", title="Планы и решения по устранению нарушений")
        fig.update_layout(xaxis_title="", yaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mgmt_actions, use_container_width=True)

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
            title="Меры юридической ответственности: принято решений и результаты",
            barmode="group"
        )
        fig.update_layout(xaxis_title="", yaxis_title="Количество", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # Показываем таблицу со всеми данными включая отмены/отказы и процент отмен
        st.caption("Детализация с процентами отмен/отказов:")
        display_df = legal_measures.copy()
        # Форматируем процент отмен
        display_df["cancellation_rate_pct"] = display_df["cancellation_rate_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
        st.dataframe(display_df, use_container_width=True)

    st.markdown("### Нормированные показатели")
    st.dataframe(normalized_activity_metrics(filtered), use_container_width=True)

with tab2:
    st.markdown("### Базовые элементы СВК")
    elems = svk_elements_summary(filtered)
    if not elems.empty:
        fig = px.bar(elems, x="element", y="yes_orgs", text="yes_orgs", title="Наличие базовых элементов СВК")
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(elems, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Фактическая форма СВК, по максимальному уровню")
        form_level = svk_form_level_summary(filtered)
        if not form_level.empty:
            fig = px.bar(form_level, x="svk_form_name", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(form_level, use_container_width=True)
    with right:
        st.markdown("### Отмеченные формы СВК")
        form_flags = svk_form_flags_summary(filtered, scoring_config)
        if not form_flags.empty:
            fig = px.bar(form_flags, x="form", y="yes_orgs", text="yes_orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(form_flags, use_container_width=True)

    st.markdown("### Рискоориентированный подход")
    risk_m = risk_methodology_summary(filtered)
    if not risk_m.empty:
        fig = px.bar(risk_m, x="category", y="orgs", text="orgs", title="Наличие и реализация методики оценки рисков")
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(risk_m, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Оценка эффективности СВК")
        eff = effectiveness_review_summary(filtered)
        if not eff.empty:
            fig = px.bar(eff, x="category", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(eff, use_container_width=True)
    with col_b:
        st.markdown("### Совершенствование СВК")
        imp = improvement_actions_summary(filtered)
        if not imp.empty:
            fig = px.bar(imp, x="category", y="orgs", text="orgs")
            fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp, use_container_width=True)

with tab3:
    st.markdown("### Покрытие активных направлений внутренним контролем")
    dsum = direction_summary(filtered, scoring_config)
    if not dsum.empty:
        chart_df = dsum.melt(
            id_vars=["direction"],
            value_vars=["covered_orgs", "uncovered_orgs"],
            var_name="status",
            value_name="orgs",
        )
        chart_df["status"] = chart_df["status"].replace({"covered_orgs": "Покрыто", "uncovered_orgs": "Не покрыто"})
        fig = px.bar(chart_df, x="direction", y="orgs", color="status", text="orgs", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dsum, use_container_width=True)

with tab4:
    st.markdown("### Отклонение от организаций с аналогичной нагрузкой")
    st.caption(
        f"Сравнение с медианой формы СВК среди организаций с похожим профилем нагрузки "
        f"(мин. группа аналогов: {peer_min_group_size} орг.; при нехватке — более грубая группировка). "
        "Отклонение — целые уровни формы."
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
            title="Распределение отклонения формы от медианы аналогов",
        )
        fig.update_layout(
            xaxis_title="Фактическая форма − медиана аналогов (уровни)",
            yaxis_title="Количество организаций",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "peer_form_median" in peer_plot.columns:
            peer_plot["peer_form_median"] = peer_plot["peer_form_median"].round().astype(int)
            fig = px.scatter(
                peer_plot,
                x="peer_form_median",
                y="svk_form_level",
                color="risk_group",
                size="peer_group_size" if "peer_group_size" in peer_plot.columns else None,
                hover_name="org_name",
                title="Фактическая форма vs медиана аналогов с аналогичной нагрузкой",
            )
            fig.update_layout(xaxis_title="Медиана формы у аналогов", yaxis_title="Фактическая форма")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Peer-метрики не найдены. Перезапустите приложение: "
            "`streamlit run app.py` из ветки `feature/peer-metrics-recommended-form`."
        )

    st.markdown("### Отклонение от расчётно рекомендуемой формы СВК")
    gap_counts = filtered.groupby("form_gap", dropna=False).size().reset_index(name="orgs").sort_values("form_gap")
    fig = px.bar(gap_counts, x="form_gap", y="orgs", text="orgs", title="Распределение отклонения от рекомендуемой формы")
    fig.update_layout(xaxis_title="Фактическая форма − рекомендуемая форма", yaxis_title="Количество организаций")
    st.plotly_chart(fig, use_container_width=True)

    scatter_cols = ["org_name", "svk_form_level", "recommended_form_level", "max_direction_load_level", "coverage_share_active", "risk_group"]
    if all(c in filtered.columns for c in scatter_cols):
        fig = px.scatter(
            filtered,
            x="recommended_form_level",
            y="svk_form_level",
            color="risk_group",
            size="max_direction_load_level",
            hover_name="org_name",
            title="Фактическая форма СВК vs расчётно рекомендуемая форма",
        )
        fig.update_layout(xaxis_title="Рекомендуемая форма", yaxis_title="Фактическая форма")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Разрез по федеральным округам")
    st.dataframe(by_dimension_summary(filtered, "federal_district"), use_container_width=True)

with tab5:
    st.markdown("### Топ организаций для управленческого внимания")
    top = top_risk_organizations(filtered, limit=100)
    st.dataframe(top, use_container_width=True)
    st.download_button(
        "Скачать топ-риск CSV",
        data=top.to_csv(index=False, encoding="utf-8-sig"),
        file_name="top_risk_organizations.csv",
        mime="text/csv",
    )

with tab6:
    st.markdown("### Организации с противоречиями в данных")
    st.caption("Логические несоответствия, требующие проверки и исправления")
    contradictions = contradictions_table(filtered)
    if not contradictions.empty:
        st.dataframe(contradictions, use_container_width=True)
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
        st.dataframe(anomalies, use_container_width=True)
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
    
    st.markdown("### Статус отчетности")
    status = report_status_summary(filtered)
    if not status.empty:
        fig = px.bar(status, x="status", y="orgs", text="orgs", title="Распределение статусов отчета")
        fig.update_layout(xaxis_title="", yaxis_title="Количество организаций")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(status, use_container_width=True)

with tab7:
    st.markdown("### Найденные колонки")
    st.caption("Если в новом годовом отчете часть колонок не найдена, дополните регулярные выражения в config/columns.yml.")
    st.dataframe(resolution_report, use_container_width=True)
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
            "среднесписочной численности и количества фактов ФХЖ; цвет — форма СВК "
            "(вид организации внутреннего контроля). Из-за сильного разброса большинство "
            "точек концентрируется у начала координат — это ожидаемо для «сырых» данных."
        )
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
        x_title = "Кассовые поступления, lg(руб.)" if log_cash else "Кассовые поступления, руб."
        y_title = "Среднесписочная численность, lg(чел.)" if log_staff else "Среднесписочная численность"
        z_title = "Количество фактов ФХЖ, lg(ед.)" if log_fkhz else "Количество фактов ФХЖ"
        raw3d = px.scatter_3d(
            plot_raw,
            x="x_cash",
            y="y_staff",
            z="z_fkhz",
            color="svk_form_name",
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
            title="Кассовые поступления × численность × факты ФХЖ (по форме СВК)",
        )
        raw3d.update_layout(
            legend_title="Форма СВК",
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
            st.dataframe(extreme[ext_cols], use_container_width=True)
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
        st.markdown("### Профиль масштаба: концентрация и форма СВК")
        st.caption(
            "Ось масштаба — направление роста по трём показателям ФХД в ln(1+x)-пространстве "
            "(первая главная компонента). Ноль — центр облака (типичный масштаб выборки); "
            "отрицательные значения — мельче типичного, положительные — крупнее. "
            "Графики показывают, где сосредоточены организации и как меняется медиана формы СВК по масштабу. "
            "Организации без положительных значений по всем трём осям исключаются из расчёта оси и бинов."
        )

        sp = scale_axis_profile(filtered, scoring_config)
        sp_bins_count = sp.get("bins_count", sp.get("bins", pd.DataFrame()))
        sp_bins_share = sp.get("bins_share", sp.get("bins", pd.DataFrame()))
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

            if not sp_active.empty:
                center_cash = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["cash_receipts"], errors="coerce").fillna(0)).mean()
                )
                center_staff = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["staff_avg"], errors="coerce").fillna(0)).mean()
                )
                center_fkhz = np.expm1(
                    np.log1p(pd.to_numeric(sp_active["fkhz_count"], errors="coerce").fillna(0)).mean()
                )

                k1, k2, k3 = st.columns(3)
                with k1:
                    metric_card("Типичные поступления", f"≈ {center_cash / 1e6:,.0f} млн руб.")
                with k2:
                    metric_card("Типичная численность", f"≈ {center_staff:,.0f} чел.")
                with k3:
                    metric_card("Типичные факты ФХЖ", f"≈ {center_fkhz:,.0f} ед.")

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
                center_idx = int(np.argmin(np.abs(bins_df["signed_scale_mid"].to_numpy())))
                return x_pos, x_axis, center_idx

            if not sp_bins_count.empty:
                x_pos_count, x_axis_count, center_idx_count = _scale_x_axis(sp_bins_count)
                bar_opacity = [0.5 if low else 1.0 for low in sp_bins_count["low_n"]]
                bar_hover = [
                    (
                        f"Медиана формы: {m:.1f}<br>Организаций: {n}"
                        + ("<br><b>n мало</b>" if low else "")
                    )
                    for m, n, low in zip(
                        sp_bins_count["form_median"], sp_bins_count["n"], sp_bins_count["low_n"]
                    )
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
                                title="Медиана формы СВК",
                                tickvals=list(range(5)),
                                ticktext=[_form_name(i) for i in range(5)],
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
                    x=center_idx_count,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Центр облака",
                    annotation_position="top",
                )
                fig_scale_n.update_layout(
                    title="Распределение организаций по масштабу",
                    xaxis=dict(title="Позиция по оси масштаба", **x_axis_count),
                    yaxis_title="Число организаций",
                    height=480,
                )
                st.plotly_chart(fig_scale_n, use_container_width=True)

            if not sp_bins_share.empty:
                x_pos_share, x_axis_share, center_idx_share = _scale_x_axis(sp_bins_share)
                fig_scale_share = go.Figure()
                for level in range(5):
                    share_pct = sp_bins_share[f"form_share_{level}"] * 100
                    count_in_bin = (sp_bins_share[f"form_share_{level}"] * sp_bins_share["n"]).round(0)
                    fig_scale_share.add_trace(
                        go.Bar(
                            name=f"{level} — {_form_name(level)}",
                            x=x_pos_share,
                            y=share_pct,
                            marker_color=_FORM_COLORS[level],
                            customdata=count_in_bin,
                            hovertemplate=(
                                f"Форма {level}<br>"
                                "Доля: %{y:.1f}%<br>"
                                "Организаций: %{customdata:.0f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                fig_scale_share.add_vline(
                    x=center_idx_share,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="gray",
                )
                fig_scale_share.update_layout(
                    barmode="stack",
                    title="Структура форм СВК по уровням масштаба",
                    xaxis=dict(title="Позиция по оси масштаба", **x_axis_share),
                    yaxis_title="Доля организаций, %",
                    yaxis=dict(range=[0, 100]),
                    legend_title="Форма СВК",
                    height=480,
                )
                st.plotly_chart(fig_scale_share, use_container_width=True)

        st.markdown("---")

        st.markdown("### Сравнение с аналогами в облаке")
        st.caption(
            f"Для каждой организации находятся {int(scoring_config.get('peer_benchmark', {}).get('min_group_size', 5))} "
            "ближайших соседей в пространстве ln(1+поступления), ln(1+численность), ln(1+ФХЖ). "
            "Сравнивается фактическая форма СВК с медианой формы у соседей. "
            "Перекос профиля показывает, по какой оси масштаб отличается от типичного у соседей; "
            "доверие к сравнению выше, когда соседи близко и их достаточно."
        )

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
                metric_card("Форма ниже соседей (надёжно)", fmt_num(len(below_peer)))
            with c3:
                metric_card("Среднее доверие к сравнению", fmt_num(avg_conf, 2) if pd.notna(avg_conf) else "—")

            plot_cp = cp.copy()
            plot_cp["log_cash"] = np.log1p(plot_cp["cash_receipts"])
            plot_cp["log_staff"] = np.log1p(plot_cp["staff_avg"])
            plot_cp["log_fkhz"] = np.log1p(plot_cp["fkhz_count"])

            st.markdown("#### 3D-облако: отклонение формы от соседей")
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
                title="Отклонение формы СВК от медианы ближайших соседей",
            )
            fig3d.update_layout(
                coloraxis_colorbar_title="Форма − медиана соседей",
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

            st.markdown("#### Фактическая форма vs медиана соседей")
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
                    title="Фактическая форма vs медиана ближайших соседей",
                )
                fig_peer.update_layout(
                    xaxis_title="Медиана формы у соседей (0–4)",
                    yaxis_title="Фактическая форма СВК (0–4)",
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

            st.markdown("#### Организации с формой ниже соседей")
            priority_tab = cloud_peer_benchmark_table(filtered, scoring_config)
            if priority_tab.empty:
                st.success("В текущей выборке приоритетных отклонений не обнаружено.")
            else:
                st.dataframe(priority_tab, use_container_width=True)
                st.download_button(
                    "Скачать peer-3D CSV",
                    data=cp_full.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="cloud_peer_3d.csv",
                    mime="text/csv",
                )
                st.info(f"Найдено организаций с формой ниже соседей (надёжное сравнение): {len(priority_tab)}")
