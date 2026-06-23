from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.svk_analytics.columns import build_canonical_frame, resolve_columns
from src.svk_analytics.io import find_latest_raw_file, load_report, load_yaml
from src.svk_analytics.scoring import enrich_with_metrics
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
    proportionality_anomalies,
    proportionality_anomalies_table,
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


def load_and_score(path: str, year: int | None):
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else None
    canonical, resolution_report = load_canonical_report(path, mtime)
    scoring_config = load_yaml("config/scoring.yml")
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
    df, scoring_config, resolution_report = load_and_score(input_path, int(year))
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
        "Сравнение с медианой формы СВК среди организаций с похожим профилем нагрузки "
        "(при нехватке аналогов — более грубая группировка). Отклонение — целые уровни формы."
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

        st.markdown("### Аномалии и соразмерность по трём осям")
        st.caption(
            "Совместный анализ суммы кассовых поступлений, среднесписочной численности и "
            "количества фактов ФХЖ с учётом формы СВК (вида организации внутреннего контроля). "
            "Аномалии — это нетипичные сочетания объёмов (многомерный выброс), непропорциональные "
            "удельные отношения и рассогласование масштаба деятельности с формой СВК."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Организаций в анализе", fmt_num(len(pa)))
        with c2:
            metric_card("С признаками аномалий", fmt_num(int(pa["is_anomaly"].sum())))
        with c3:
            metric_card("Многомерных выбросов", fmt_num(int(pa["scale_outlier"].sum())))

        st.markdown("#### Трёхмерная карта организаций")
        st.caption(
            "Оси — в логарифмическом масштабе ln(1+x). Цвет — форма СВК, размер маркера — "
            "итоговый балл аномальности. Наведите курсор для деталей по организации."
        )
        plot_df = kept.copy()
        plot_df["log_cash"] = np.log1p(plot_df["cash_receipts"])
        plot_df["log_staff"] = np.log1p(plot_df["staff_avg"])
        plot_df["log_fkhz"] = np.log1p(plot_df["fkhz_count"])
        plot_df["Балл аномальности"] = plot_df["anomaly_score"].fillna(0)
        plot_df["marker_size"] = np.sqrt(plot_df["anomaly_score"].fillna(0)) + 1.0

        fig3d = px.scatter_3d(
            plot_df,
            x="log_cash",
            y="log_staff",
            z="log_fkhz",
            color="svk_form_name",
            size="marker_size",
            size_max=22,
            opacity=0.75,
            hover_name="org_name",
            hover_data={
                "log_cash": False,
                "log_staff": False,
                "log_fkhz": False,
                "marker_size": False,
                "cash_receipts": ":,.0f",
                "staff_avg": ":,.0f",
                "fkhz_count": ":,.0f",
                "scale_level": True,
                "scale_vs_form": True,
                "Балл аномальности": ":.1f",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Кассовые поступления × численность × факты ФХЖ (по форме СВК)",
        )
        fig3d.update_layout(
            legend_title="Форма СВК",
            scene=dict(
                xaxis_title="ln(1+кассовые поступления)",
                yaxis_title="ln(1+численность)",
                zaxis_title="ln(1+факты ФХЖ)",
            ),
            height=700,
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("#### Масштаб деятельности и форма СВК")
        st.caption(
            "Совокупный масштаб (0–4) против формы СВК (0–4). Точки ниже диагонали — крупные "
            "организации со сравнительно слабой формой; выше — развитая форма при малом масштабе."
        )
        mismatch_df = (
            pa.groupby(["scale_level", "svk_form_level"]).size().reset_index(name="orgs")
        )
        fig_mm = px.scatter(
            mismatch_df,
            x="scale_level",
            y="svk_form_level",
            size="orgs",
            color="orgs",
            color_continuous_scale="Blues",
            title="Распределение организаций: масштаб × форма СВК",
        )
        fig_mm.update_layout(
            xaxis_title="Совокупный масштаб (0–4)",
            yaxis_title="Форма СВК (0–4)",
        )
        st.plotly_chart(fig_mm, use_container_width=True)

        st.markdown("#### Организации с признаками аномалий")
        anomaly_tab = proportionality_anomalies_table(filtered, scoring_config)
        if anomaly_tab.empty:
            st.success("В текущей выборке аномалий не обнаружено.")
        else:
            st.dataframe(anomaly_tab, use_container_width=True)
            st.download_button(
                "Скачать аномалии CSV",
                data=anomaly_tab.to_csv(index=False, encoding="utf-8-sig"),
                file_name="proportionality_anomalies.csv",
                mime="text/csv",
            )
            st.info(f"Найдено организаций с признаками аномалий: {len(anomaly_tab)}")
