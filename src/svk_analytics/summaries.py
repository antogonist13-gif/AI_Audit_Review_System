from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cleaning import clean_text, to_number, yes_flag


def _filled_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with filled report. If status column is absent, all rows are treated as filled."""
    if "is_report_filled" in df.columns:
        mask = df["is_report_filled"].fillna(True).astype(bool)
        return df.loc[mask].copy()
    return df.copy()


def _safe_yes(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float")
    return yes_flag(df[col])


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float")
    return to_number(df[col])


def _pct(numerator: float, denominator: float) -> float | None:
    if denominator in (0, None) or pd.isna(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def _money_bln(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) / 1_000_000_000


def _classify_risk_methodology(value) -> str:
    text = (clean_text(value) or "").lower()
    if not text:
        return "Не указано"
    if "отсутств" in text or "не применяется" in text:
        return "Методика отсутствует"
    if "фрагментар" in text or "несистем" in text:
        return "Фрагментарная / несистемная"
    if "утвержден" in text or "карты рисков" in text or "риск" in text and "план" in text:
        return "Полноценная методика"
    return clean_text(value) or "Не указано"


def _classify_effectiveness_review(value) -> str:
    text = (clean_text(value) or "").lower()
    if not text:
        return "Не указано"
    if "не проводится" in text:
        return "Оценка не проводится"
    if "нерегуляр" in text or "не оформ" in text:
        return "Нерегулярная / без оформления"
    if "регуляр" in text or "не реже" in text:
        return "Регулярная оценка"
    return clean_text(value) or "Не указано"


def _classify_improvement_actions(value) -> str:
    text = (clean_text(value) or "").lower()
    if not text:
        return "Не указано"
    if "не планируются" in text or "не проводятся" in text:
        return "Не планируются / не проводятся"
    if "не в полном объеме" in text or "планируются" in text:
        return "Планируются / частично выполняются"
    if "разрабатываются" in text or "реализуются" in text or "документирован" in text:
        return "Документированные мероприятия реализуются"
    return clean_text(value) or "Не указано"


def report_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "report_status" not in df.columns:
        return pd.DataFrame()
    s = df["report_status"].map(clean_text).fillna("Не указано")
    out = s.value_counts(dropna=False).rename_axis("status").reset_index(name="orgs")
    out["percent"] = (out["orgs"] / len(df) * 100) if len(df) else np.nan
    return out


def svk_elements_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    elements = [
        ("svk_organized", "СВК организован и осуществляется"),
        ("accounting_policy", "Раздел СВК в учетной политике"),
        ("lna_approved", "Организация СВК утверждена ЛНА"),
        ("powers_defined", "Определены полномочия ответственных лиц"),
        ("plan_schedule", "План-график мероприятий СВК"),
    ]
    rows = []
    denom = len(base)
    for col, label in elements:
        values = _safe_yes(base, col)
        yes = int((values == 1).sum())
        no = int((values == 0).sum())
        missing = int(values.isna().sum())
        pct_value = _pct(yes, denom)
        rows.append(
            {
                "element_key": col,
                "element": label,
                "yes_orgs": yes,
                "no_orgs": no,
                "missing_orgs": missing,
                "filled_reports": denom,
                "yes_rate_pct": pct_value * 100 if pct_value is not None else None,
            }
        )
    return pd.DataFrame(rows)


def svk_full_basic_package_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    required_cols = ["accounting_policy", "lna_approved", "powers_defined", "plan_schedule"]
    available_cols = [c for c in required_cols if c in base.columns]
    if not available_cols:
        return pd.DataFrame()
    mask = pd.Series(True, index=base.index)
    for col in available_cols:
        mask = mask & (_safe_yes(base, col) == 1)
    count = int(mask.sum())
    pct_value = _pct(count, len(base))
    return pd.DataFrame(
        [
            {
                "metric": "Организаций с полным базовым комплектом СВК",
                "value": count,
                "denominator": len(base),
                "percent": pct_value * 100 if pct_value is not None else None,
                "included_elements": ", ".join(available_cols),
            }
        ]
    )


def svk_form_flags_summary(df: pd.DataFrame, scoring_config: dict[str, Any] | None = None) -> pd.DataFrame:
    base = _filled_df(df)
    default_forms = {
        "form_responsible_person": {"level": 1, "name": "Уполномоченное должностное лицо"},
        "form_temporary_body": {"level": 2, "name": "Временный коллегиальный орган / комиссия"},
        "form_permanent_body": {"level": 3, "name": "Постоянно действующий коллегиальный орган"},
        "form_structural_unit": {"level": 4, "name": "Уполномоченное структурное подразделение"},
    }
    forms = (scoring_config or {}).get("forms", default_forms)
    rows = []
    for col, spec in forms.items():
        values = _safe_yes(base, col)
        yes = int((values == 1).sum())
        pct_value = _pct(yes, len(base))
        rows.append(
            {
                "form_key": col,
                "form": spec.get("name", col),
                "level": int(spec.get("level", 0)),
                "yes_orgs": yes,
                "filled_reports": len(base),
                "percent_filled": pct_value * 100 if pct_value is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values("level")


def svk_form_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    if "svk_form_level" not in base.columns:
        return pd.DataFrame()
    out = (
        base.groupby(["svk_form_level", "svk_form_name"], dropna=False)
        .size()
        .reset_index(name="orgs")
        .sort_values("svk_form_level")
    )
    out["percent_filled"] = (out["orgs"] / len(base) * 100) if len(base) else np.nan
    return out


def _classified_summary(df: pd.DataFrame, column: str, classifier: Callable[[Any], str]) -> pd.DataFrame:
    base = _filled_df(df)
    if column not in base.columns:
        return pd.DataFrame()
    s = base[column].map(classifier)
    out = s.value_counts(dropna=False).rename_axis("category").reset_index(name="orgs")
    out["percent_filled"] = (out["orgs"] / len(base) * 100) if len(base) else np.nan
    return out


def risk_methodology_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _classified_summary(df, "risk_methodology", _classify_risk_methodology)


def effectiveness_review_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _classified_summary(df, "effectiveness_review", _classify_effectiveness_review)


def improvement_actions_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _classified_summary(df, "improvement_actions", _classify_improvement_actions)


def violations_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    total = _safe_num(base, "violations_total").fillna(0)
    fixed_on_time = _safe_num(base, "violations_fixed_on_time").fillna(0)
    fixed_late = _safe_num(base, "violations_fixed_late").fillna(0)
    fixed_total = fixed_on_time + fixed_late
    unresolved_by_row = (total - fixed_total).clip(lower=0)
    total_sum = float(total.sum())

    rows = [
        {
            "metric": "Выявленные нарушения",
            "value": total_sum,
            "share_of_violations": 1.0 if total_sum else None,
            "per_100_violations": 100.0 if total_sum else None,
        },
        {
            "metric": "Устранено в срок",
            "value": float(fixed_on_time.sum()),
            "share_of_violations": _pct(fixed_on_time.sum(), total_sum),
            "per_100_violations": _pct(fixed_on_time.sum(), total_sum) * 100 if total_sum else None,
        },
        {
            "metric": "Устранено с нарушением срока",
            "value": float(fixed_late.sum()),
            "share_of_violations": _pct(fixed_late.sum(), total_sum),
            "per_100_violations": _pct(fixed_late.sum(), total_sum) * 100 if total_sum else None,
        },
        {
            "metric": "Остаток без отраженного устранения",
            "value": float(unresolved_by_row.sum()),
            "share_of_violations": _pct(unresolved_by_row.sum(), total_sum),
            "per_100_violations": _pct(unresolved_by_row.sum(), total_sum) * 100 if total_sum else None,
        },
    ]

    action_cols = [
        ("remediation_plans", "Планы устранения нарушений"),
        ("lna_changes", "Решения по внесению изменений в ЛНА"),
        ("disciplinary_decisions", "Решения о дисциплинарной ответственности"),
        ("materials_law_enforcement", "Материалы в правоохранительные / контрольные органы"),
        ("materials_court", "Материалы в суд"),
    ]
    for col, label in action_cols:
        value = float(_safe_num(base, col).fillna(0).sum())
        rows.append(
            {
                "metric": label,
                "value": value,
                "share_of_violations": _pct(value, total_sum),
                "per_100_violations": _pct(value, total_sum) * 100 if total_sum else None,
            }
        )

    rows.append(
        {
            "metric": "Строк с арифметическим противоречием: устранено больше выявлено",
            "value": int(((total > 0) & (fixed_total > total)).sum()),
            "share_of_violations": None,
            "per_100_violations": None,
        }
    )
    return pd.DataFrame(rows)


def violations_and_remediation(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of violations and their remediation status."""
    base = _filled_df(df)
    total = _safe_num(base, "violations_total").fillna(0)
    fixed_on_time = _safe_num(base, "violations_fixed_on_time").fillna(0)
    fixed_late = _safe_num(base, "violations_fixed_late").fillna(0)
    fixed_total = fixed_on_time + fixed_late
    unresolved_by_row = (total - fixed_total).clip(lower=0)
    total_sum = float(total.sum())

    rows = [
        {
            "metric": "Выявленные нарушения",
            "value": total_sum,
            "percent_of_violations": 100.0 if total_sum else None,
        },
        {
            "metric": "Устранено в срок",
            "value": float(fixed_on_time.sum()),
            "percent_of_violations": _pct(fixed_on_time.sum(), total_sum) * 100 if _pct(fixed_on_time.sum(), total_sum) is not None else None,
        },
        {
            "metric": "Устранено с нарушением срока",
            "value": float(fixed_late.sum()),
            "percent_of_violations": _pct(fixed_late.sum(), total_sum) * 100 if _pct(fixed_late.sum(), total_sum) is not None else None,
        },
        {
            "metric": "Остаток без отраженного устранения",
            "value": float(unresolved_by_row.sum()),
            "percent_of_violations": _pct(unresolved_by_row.sum(), total_sum) * 100 if _pct(unresolved_by_row.sum(), total_sum) is not None else None,
        },
    ]
    return pd.DataFrame(rows)


def management_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of management actions (plans and decisions)."""
    base = _filled_df(df)
    total = _safe_num(base, "violations_total").fillna(0)
    total_sum = float(total.sum())

    action_cols = [
        ("remediation_plans", "Планы устранения нарушений"),
        ("lna_changes", "Решения по внесению изменений в ЛНА"),
    ]
    
    rows = []
    for col, label in action_cols:
        value = float(_safe_num(base, col).fillna(0).sum())
        pct_value = _pct(value, total_sum)
        rows.append(
            {
                "metric": label,
                "value": value,
                "percent_of_violations": pct_value * 100 if pct_value is not None else None,
            }
        )
    return pd.DataFrame(rows)


def legal_responsibility_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of legal responsibility measures with cancellation/refusal details."""
    base = _filled_df(df)
    
    # Основные меры и соответствующие им отмены/отказы
    disciplinary = float(_safe_num(base, "disciplinary_decisions").fillna(0).sum())
    cancelled = float(_safe_num(base, "cancelled_decisions").fillna(0).sum())
    
    law_enforcement = float(_safe_num(base, "materials_law_enforcement").fillna(0).sum())
    returned = float(_safe_num(base, "returned_refused").fillna(0).sum())
    
    court = float(_safe_num(base, "materials_court").fillna(0).sum())
    refused = float(_safe_num(base, "refused_satisfaction").fillna(0).sum())
    
    rows = [
        {
            "metric": "Дисциплинарная ответственность: решений принято",
            "value": disciplinary,
            "cancelled_returned_refused": None,
            "cancellation_rate_pct": None,
        },
        {
            "metric": "  → из них отменены в порядке обжалования",
            "value": cancelled,
            "cancelled_returned_refused": cancelled,
            "cancellation_rate_pct": _pct(cancelled, disciplinary) * 100 if disciplinary else None,
        },
        {
            "metric": "Материалы в правоохранительные органы: направлено",
            "value": law_enforcement,
            "cancelled_returned_refused": None,
            "cancellation_rate_pct": None,
        },
        {
            "metric": "  → из них возвращены, получен отказ",
            "value": returned,
            "cancelled_returned_refused": returned,
            "cancellation_rate_pct": _pct(returned, law_enforcement) * 100 if law_enforcement else None,
        },
        {
            "metric": "Материалы в суд: направлено",
            "value": court,
            "cancelled_returned_refused": None,
            "cancellation_rate_pct": None,
        },
        {
            "metric": "  → из них отказано в удовлетворении",
            "value": refused,
            "cancelled_returned_refused": refused,
            "cancellation_rate_pct": _pct(refused, court) * 100 if court else None,
        },
    ]
    return pd.DataFrame(rows)


def normalized_activity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    violations = float(_safe_num(base, "violations_total").fillna(0).sum())
    staff = float(_safe_num(base, "staff_avg").fillna(0).sum())
    cash_payments = float(_safe_num(base, "cash_payments").fillna(0).sum())
    fkhz = float(_safe_num(base, "fkhz_count").fillna(0).sum())
    procurements = float(_safe_num(base, "procurement_count").fillna(0).sum())
    procurement_amount = float(_safe_num(base, "procurement_amount").fillna(0).sum())

    rows = [
        {"metric": "Среднесписочная численность, всего", "value": staff, "unit": "чел."},
        {"metric": "Кассовые выплаты, всего", "value": cash_payments, "unit": "руб."},
        {"metric": "Кассовые выплаты, всего", "value": _money_bln(cash_payments), "unit": "млрд руб."},
        {"metric": "Нарушений на 100 среднесписочных сотрудников", "value": violations / staff * 100 if staff else None, "unit": "ед."},
        {"metric": "Нарушений на 1 млрд руб. кассовых выплат", "value": violations / (cash_payments / 1_000_000_000) if cash_payments else None, "unit": "ед."},
        {"metric": "Нарушений на 1 000 фактов хозяйственной жизни", "value": violations / fkhz * 1000 if fkhz else None, "unit": "ед."},
        {"metric": "Закупок на 100 сотрудников", "value": procurements / staff * 100 if staff else None, "unit": "ед."},
        {"metric": "Совокупный объем закупок", "value": procurement_amount, "unit": "руб."},
        {"metric": "Совокупный объем закупок", "value": _money_bln(procurement_amount), "unit": "млрд руб."},
        {"metric": "Доля объема закупок в кассовых выплатах", "value": (procurement_amount / cash_payments * 100) if cash_payments else None, "unit": "%"},
    ]
    return pd.DataFrame(rows)


def overview_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base = _filled_df(df)
    n = len(df)
    filled = len(base)
    svk_org = int((_safe_yes(base, "svk_organized") == 1).sum()) if "svk_organized" in base.columns else None
    full_basic = svk_full_basic_package_summary(df)
    full_basic_count = int(full_basic["value"].iloc[0]) if not full_basic.empty else None

    risk_summary = risk_methodology_summary(df)
    full_risk = int(risk_summary.loc[risk_summary["category"] == "Полноценная методика", "orgs"].sum()) if not risk_summary.empty else None
    effectiveness = effectiveness_review_summary(df)
    regular_eff = int(effectiveness.loc[effectiveness["category"] == "Регулярная оценка", "orgs"].sum()) if not effectiveness.empty else None

    viol = violations_summary(df)
    total_violations = float(viol.loc[viol["metric"] == "Выявленные нарушения", "value"].iloc[0]) if not viol.empty else None
    fixed_on_time_pct = float(viol.loc[viol["metric"] == "Устранено в срок", "share_of_violations"].iloc[0]) if not viol.empty else None
    # Конвертируем в проценты если это доля
    if fixed_on_time_pct is not None and fixed_on_time_pct <= 1.0:
        fixed_on_time_pct = fixed_on_time_pct * 100

    pct_filled = _pct(filled, n)
    pct_svk_org = _pct(svk_org, filled) if svk_org is not None else None
    pct_full_basic = _pct(full_basic_count, filled) if full_basic_count is not None else None
    pct_full_risk = _pct(full_risk, filled) if full_risk is not None else None
    pct_regular_eff = _pct(regular_eff, filled) if regular_eff is not None else None
    avg_coverage = df["coverage_share_active"].mean() if "coverage_share_active" in df.columns else None
    
    rows = [
        ("Организаций в отчете", n),
        ("Заполненных отчетов", filled),
        ("% заполненных отчетов", pct_filled * 100 if pct_filled is not None else None),
        ("Организаций, где СВК организован", svk_org),
        ("% организаций, где СВК организован", pct_svk_org * 100 if pct_svk_org is not None else None),
        ("Организаций с полным базовым комплектом СВК", full_basic_count),
        ("% организаций с полным базовым комплектом СВК", pct_full_basic * 100 if pct_full_basic is not None else None),
        ("Организаций с полноценной методикой оценки рисков", full_risk),
        ("% организаций с полноценной методикой оценки рисков", pct_full_risk * 100 if pct_full_risk is not None else None),
        ("Организаций с регулярной оценкой эффективности СВК", regular_eff),
        ("% организаций с регулярной оценкой эффективности СВК", pct_regular_eff * 100 if pct_regular_eff is not None else None),
        ("Выявленные нарушения", total_violations),
        ("% нарушений, устраненных в срок", fixed_on_time_pct),
        ("Среднее покрытие активных направлений, %", avg_coverage * 100 if avg_coverage is not None else None),
        ("Организаций с непокрытыми активными направлениями", int((df["uncovered_active_directions_count"] > 0).sum()) if "uncovered_active_directions_count" in df.columns else None),
        ("Организаций со слабой формой СВК", int((df["form_gap"] < 0).sum()) if "form_gap" in df.columns else None),
        ("Организаций с двойным риском", int(df["risk_group"].str.startswith("D.").sum()) if "risk_group" in df.columns else None),
        ("Организаций с флагами качества данных", int((df["data_quality_flags"].fillna("") != "").sum()) if "data_quality_flags" in df.columns else None),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def direction_summary(df: pd.DataFrame, scoring_config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, spec in scoring_config["directions"].items():
        active = df[f"{key}_active"] == 1
        covered = df[f"{key}_covered"] == 1
        uncovered = active & ~covered
        coverage = (active & covered).sum() / active.sum() if active.sum() else None
        rows.append(
            {
                "direction_key": key,
                "direction": spec["name"],
                "active_orgs": int(active.sum()),
                "covered_orgs": int((active & covered).sum()),
                "uncovered_orgs": int(uncovered.sum()),
                "coverage_pct_active": float(coverage * 100) if coverage is not None else None,
                "high_load_orgs": int((df[f"{key}_load_level"] >= 3).sum()),
                "high_load_uncovered_orgs": int(((df[f"{key}_load_level"] >= 3) & uncovered).sum()),
                "avg_load_level": float(df[f"{key}_load_level"].mean()),
            }
        )
    return pd.DataFrame(rows)


def risk_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("risk_group", dropna=False)
        .size()
        .reset_index(name="orgs")
        .assign(percent=lambda x: x["orgs"] / x["orgs"].sum() * 100)
        .sort_values("risk_group")
    )


def by_dimension_summary(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(dimension, dropna=False)
        .agg(
            orgs=("org_name", "count") if "org_name" in df.columns else ("risk_group", "count"),
            avg_coverage=("coverage_share_active", "mean"),
            avg_form_gap=("form_gap", "mean"),
            double_risk_orgs=("risk_group", lambda s: int(s.str.startswith("D.").sum())),
            weak_form_orgs=("form_gap", lambda s: int((s < 0).sum())),
            uncovered_orgs=("uncovered_active_directions_count", lambda s: int((s > 0).sum())),
        )
        .reset_index()
        .sort_values(["double_risk_orgs", "weak_form_orgs", "uncovered_orgs"], ascending=False)
    )


def data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    if "data_quality_flags" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["org_name", "org_type", "federal_district", "region", "report_status", "data_quality_flags"] if c in df.columns]
    return df.loc[df["data_quality_flags"].fillna("") != "", cols].copy()


def contradictions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return organizations with logical contradictions in data."""
    if "contradictions" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["org_name", "federal_district", "region", 
                        "contact_name", "contact_phone", "contact_email",
                        "contradictions"] if c in df.columns]
    return df.loc[df["contradictions"].fillna("") != "", cols].copy()


def anomalies_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return organizations with suspicious data patterns (anomalies)."""
    if "anomalies" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["org_name", "federal_district", "region",
                        "contact_name", "contact_phone", "contact_email",
                        "anomalies"] if c in df.columns]
    return df.loc[df["anomalies"].fillna("") != "", cols].copy()


def top_risk_organizations(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    display_cols = [
        "org_name", "org_type", "federal_district", "region", "risk_group",
        "coverage_share_active", "uncovered_load_sum", "max_uncovered_load",
        "svk_form_name", "recommended_form_name", "form_gap",
        "peer_form_median", "form_vs_peer", "peer_group_size",
        "fhd_load_level", "procurement_load_level", "property_load_level", "project_load_level",
        "fhd_covered", "procurement_covered", "property_covered", "project_covered",
        "data_quality_flags",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    out = df.copy()
    out["risk_sort"] = (
        out["uncovered_load_sum"].fillna(0) * 10
        + out["max_uncovered_load"].fillna(0) * 5
        + (-out["form_gap"].clip(upper=0)).fillna(0) * 3
        + (out["data_quality_flags"].fillna("") != "").astype(int)
    )
    return out.sort_values("risk_sort", ascending=False)[display_cols].head(limit)


# --- Анализ аномалий и соразмерности по трём осям ФХД (аддитивный блок) ---------
# Оси: сумма кассовых поступлений, среднесписочная численность, количество фактов ФХЖ.
# Дополнительная ось — форма СВК (вид организации внутреннего контроля).
# Реализация без новых зависимостей (только numpy/pandas).

ANOMALY_AXES: list[tuple[str, str]] = [
    ("cash_receipts", "Сумма кассовых поступлений"),
    ("staff_avg", "Среднесписочная численность"),
    ("fkhz_count", "Количество фактов ФХЖ"),
]

# Удельные отношения для оценки соразмерности (непропорциональности) осей.
ANOMALY_RATIOS: list[tuple[str, str, str, str]] = [
    ("cash_per_staff", "cash_receipts", "staff_avg", "поступления на 1 сотрудника"),
    ("fkhz_per_staff", "fkhz_count", "staff_avg", "факты ФХЖ на 1 сотрудника"),
    ("cash_per_fkhz", "cash_receipts", "fkhz_count", "поступления на 1 факт ФХЖ"),
]

# Порог робастного z-отклонения для удельных отношений и квантиль χ²(3) для
# многомерного выброса (0.99 ≈ 11.345).
_RATIO_Z_THRESHOLD = 3.5
_MAHALANOBIS_CHI2_99 = 11.345


def _percentile_rank(values: pd.Series) -> pd.Series:
    """Percentile rank among positive values (consistent with scoring methodology)."""
    pct = pd.Series(0.0, index=values.index)
    positive = values > 0
    if positive.sum() > 0:
        pct.loc[positive] = values.loc[positive].rank(pct=True)
    return pct


def _scale_level_from_score(score: pd.Series, thresholds: dict[str, float] | None) -> pd.Series:
    """Map a 0..1 percentile-style score to a 0..4 level using load-level thresholds."""
    thresholds = thresholds or {}
    l1 = float(thresholds.get("level_1_max_percentile", 0.75))
    l2 = float(thresholds.get("level_2_max_percentile", 0.90))
    l3 = float(thresholds.get("level_3_max_percentile", 0.95))
    level = pd.Series(0, index=score.index, dtype="int64")
    level[(score > 0) & (score <= l1)] = 1
    level[(score > l1) & (score <= l2)] = 2
    level[(score > l2) & (score <= l3)] = 3
    level[score > l3] = 4
    return level


def _robust_z(log_values: pd.Series) -> pd.Series:
    """Robust z-score (median/MAD) on a log-scale series; NaN where undefined."""
    valid = log_values.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=log_values.index)
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if mad == 0:
        std = float(valid.std(ddof=0))
        if not std or pd.isna(std):
            return pd.Series(np.nan, index=log_values.index)
        return (log_values - med) / std
    return (log_values - med) / (1.4826 * mad)


def _mahalanobis_sq(coords: pd.DataFrame) -> pd.Series:
    """Squared Mahalanobis distance per row (numpy only); NaN if too few rows."""
    result = pd.Series(np.nan, index=coords.index)
    data = coords.to_numpy(dtype="float64")
    mask = ~np.isnan(data).any(axis=1)
    if mask.sum() < max(10, coords.shape[1] + 1):
        return result
    sub = data[mask]
    mean = sub.mean(axis=0)
    cov = np.cov(sub, rowvar=False)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    diff = sub - mean
    md_sq = np.einsum("ij,jk,ik->i", diff, inv, diff)
    result.loc[coords.index[mask]] = md_sq
    return result


def proportionality_anomalies(
    df: pd.DataFrame, scoring_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Per-organization anomaly and proportionality analysis across three FHD axes.

    Returns one row per filled report with:
    - raw axis values and their percentile ranks (``*_pct``);
    - совокупный масштаб (``scale_score`` 0..1, ``scale_level`` 0..4);
    - форма СВК (``svk_form_level``/``svk_form_name``);
    - многомерный выброс (``mahalanobis``, ``scale_outlier``);
    - удельные отношения и их робастные z-отклонения (``*_z``);
    - рассогласование масштаб↔форма (``scale_vs_form``);
    - итоговый балл (``anomaly_score``), текстовые причины (``anomaly_reasons``)
      и флаг ``is_anomaly``.

    Чисто читающая, аддитивная функция: исходный DataFrame не изменяется.
    """
    base = _filled_df(df)
    if base.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=base.index)
    for col in ["org_name", "org_type", "org_type_classified", "federal_district", "region"]:
        if col in base.columns:
            out[col] = base[col]

    vals: dict[str, pd.Series] = {}
    log_coords: dict[str, pd.Series] = {}
    pct_ranks: dict[str, pd.Series] = {}
    for col, _label in ANOMALY_AXES:
        s = _safe_num(base, col).clip(lower=0)
        vals[col] = s
        out[col] = s
        log_coords[col] = np.log1p(s)
        pct = _percentile_rank(s)
        pct_ranks[col] = pct
        out[f"{col}_pct"] = (pct * 100).round(1)

    # Совокупный масштаб: средний перцентиль по трём осям, уровень — по рангу масштаба.
    scale_score = pd.concat(pct_ranks.values(), axis=1).mean(axis=1)
    out["scale_score"] = scale_score.round(3)
    scale_rank = scale_score.rank(pct=True)
    thresholds = (scoring_config or {}).get("load_level_thresholds", {})
    out["scale_level"] = _scale_level_from_score(scale_rank, thresholds)

    if "svk_form_level" in base.columns:
        out["svk_form_level"] = base["svk_form_level"].fillna(0).astype(int)
    else:
        out["svk_form_level"] = 0
    if "svk_form_name" in base.columns:
        out["svk_form_name"] = base["svk_form_name"]
    else:
        out["svk_form_name"] = out["svk_form_level"].astype(str)

    # Многомерный выброс по лог-координатам (учитывает корреляцию осей).
    coords = pd.concat([log_coords[c].rename(c) for c, _ in ANOMALY_AXES], axis=1)
    all_zero = pd.concat(vals.values(), axis=1).sum(axis=1) == 0
    md_sq = _mahalanobis_sq(coords.mask(all_zero))
    out["mahalanobis"] = np.sqrt(md_sq).round(2)
    out["scale_outlier"] = (md_sq > _MAHALANOBIS_CHI2_99).fillna(False)

    # Удельные отношения и робастные z-отклонения в лог-шкале.
    reason_lists: list[list[str]] = [[] for _ in range(len(base))]
    z_abs_cols: list[pd.Series] = []
    for name, num_col, den_col, label in ANOMALY_RATIOS:
        num = vals[num_col]
        den = vals[den_col]
        ratio = (num / den.replace(0, np.nan)).where(num > 0)
        out[name] = ratio
        z = _robust_z(np.log(ratio.where(ratio > 0)))
        out[f"{name}_z"] = z.round(2)
        z_abs_cols.append(z.abs())
        for i, idx in enumerate(base.index):
            zi = z.get(idx, np.nan)
            if pd.isna(zi):
                continue
            if zi > _RATIO_Z_THRESHOLD:
                reason_lists[i].append(f"Аномально высокое: {label}")
            elif zi < -_RATIO_Z_THRESHOLD:
                reason_lists[i].append(f"Аномально низкое: {label}")

    out["scale_vs_form"] = out["scale_level"] - out["svk_form_level"]

    md_flag = out["scale_outlier"].to_numpy()
    scale_lvl = out["scale_level"].to_numpy()
    form_lvl = out["svk_form_level"].to_numpy()
    for i in range(len(base)):
        if md_flag[i]:
            reason_lists[i].insert(0, "Нетипичное сочетание объёмов (многомерный выброс)")
        if scale_lvl[i] >= 3 and form_lvl[i] <= 1:
            reason_lists[i].append("Крупный масштаб при слабой форме СВК")
        elif scale_lvl[i] <= 1 and form_lvl[i] >= 4:
            reason_lists[i].append("Развитая форма СВК при малом масштабе")

    out["anomaly_reasons"] = ["; ".join(r) for r in reason_lists]
    out["is_anomaly"] = out["anomaly_reasons"].fillna("") != ""

    md_norm = np.sqrt(md_sq).fillna(0.0)
    z_max = (
        pd.concat(z_abs_cols, axis=1).max(axis=1).fillna(0.0)
        if z_abs_cols
        else pd.Series(0.0, index=base.index)
    )
    mismatch = out["scale_vs_form"].abs().fillna(0)
    out["anomaly_score"] = (md_norm + z_max + mismatch).round(2)

    return out.reset_index(drop=True)


def proportionality_anomalies_table(
    df: pd.DataFrame, scoring_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Compact table of flagged organizations only, sorted by anomaly score."""
    pa = proportionality_anomalies(df, scoring_config)
    if pa.empty:
        return pa
    flagged = pa[pa["is_anomaly"]].copy()
    cols = [
        c
        for c in [
            "org_name", "org_type_classified", "federal_district", "region",
            "cash_receipts", "staff_avg", "fkhz_count",
            "scale_level", "svk_form_name", "scale_vs_form",
            "mahalanobis", "anomaly_score", "anomaly_reasons",
        ]
        if c in flagged.columns
    ]
    return flagged.sort_values("anomaly_score", ascending=False)[cols].reset_index(drop=True)


def split_extreme_values(
    pa: pd.DataFrame, percentiles: float | dict[str, float] = 0.995
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the anomaly frame into (kept, extreme) by one-sided high percentiles.

    Организация считается экстремальной, если хотя бы по одной из трёх осей
    (кассовые поступления, численность, факты ФХЖ) её значение превышает порог
    этой оси. ``percentiles`` — либо одно число (общий перцентиль для всех осей),
    либо словарь ``{колонка: перцентиль}`` для индивидуальной настройки.
    Такие наблюдения «растягивают» 3D-график и выносятся в отдельную таблицу
    на проверку. ``extreme`` дополняется колонкой ``extreme_reason``.
    """
    if pa is None or pa.empty:
        empty = pa if pa is not None else pd.DataFrame()
        return empty, empty.iloc[0:0] if not empty.empty else empty

    def _pct_for(col: str) -> float:
        raw = percentiles.get(col, 0.995) if isinstance(percentiles, dict) else percentiles
        return min(max(float(raw), 0.0), 1.0)

    thresholds: dict[str, float] = {}
    extreme_mask = pd.Series(False, index=pa.index)
    for col, _label in ANOMALY_AXES:
        if col not in pa.columns:
            continue
        s = pd.to_numeric(pa[col], errors="coerce")
        t = s.quantile(_pct_for(col))
        thresholds[col] = t
        extreme_mask = extreme_mask | (s > t)

    def _reason(row: pd.Series) -> str:
        parts: list[str] = []
        for col, label in ANOMALY_AXES:
            t = thresholds.get(col)
            v = row.get(col)
            if t is None or pd.isna(t) or pd.isna(v) or v <= t:
                continue
            ratio = (v / t) if t else float("inf")
            parts.append(f"{label}: {v:,.0f} (×{ratio:.1f} от порога)")
        return "; ".join(parts)

    extreme = pa[extreme_mask].copy()
    if not extreme.empty:
        extreme["extreme_reason"] = extreme.apply(_reason, axis=1)
        sort_col = "anomaly_score" if "anomaly_score" in extreme.columns else None
        if sort_col:
            extreme = extreme.sort_values(sort_col, ascending=False)
    kept = pa[~extreme_mask].copy()
    return kept.reset_index(drop=True), extreme.reset_index(drop=True)
