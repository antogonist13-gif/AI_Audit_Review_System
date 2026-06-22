from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cleaning import to_number, yes_flag, clean_text


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return to_number(df[col]).fillna(0.0)


def _safe_yes(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return yes_flag(df[col]).fillna(0.0)


def _percentile_scores(values: pd.Series) -> pd.Series:
    """Return percentile rank for positive values; zero/non-positive remain zero."""
    values = values.fillna(0.0).clip(lower=0.0)
    result = pd.Series(0.0, index=values.index)
    positive = values > 0
    if positive.sum() > 0:
        result.loc[positive] = values.loc[positive].rank(method="average", pct=True)
    return result


def _level_from_percentile(score: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    l1 = float(thresholds.get("level_1_max_percentile", 0.75))
    l2 = float(thresholds.get("level_2_max_percentile", 0.90))
    l3 = float(thresholds.get("level_3_max_percentile", 0.95))

    level = pd.Series(0, index=score.index, dtype="int64")
    level[(score > 0) & (score <= l1)] = 1
    level[(score > l1) & (score <= l2)] = 2
    level[(score > l2) & (score <= l3)] = 3
    level[score > l3] = 4
    return level


def _form_name(level: int) -> str:
    return {
        0: "Иная форма",
        1: "Уполномоченное должностное лицо",
        2: "Временный коллегиальный орган / комиссия",
        3: "Постоянно действующий коллегиальный орган",
        4: "Уполномоченное структурное подразделение",
    }.get(int(level), "Неизвестно")


def classify_org_type(org_type_value: str) -> str:
    """Classify organization type based on org_type column value.
    
    Groups 8 original types into 3 categories:
    - Высшее образование (233 orgs)
    - Научные учреждения (404 orgs)
    - Другие (48 orgs: государственные учреждения, дошкольные, унитарные предприятия, СПО, культура, ДПО)
    """
    if not org_type_value or pd.isna(org_type_value):
        return "Другие"
    
    org_type_str = str(org_type_value).strip()
    
    # Высшее образование
    if org_type_str == "Высшее образование":
        return "Высшее образование"
    
    # Научные учреждения
    if org_type_str == "Научные учреждения":
        return "Научные учреждения"
    
    # Все остальные типы
    return "Другие"


def enrich_with_metrics(df: pd.DataFrame, scoring_config: dict[str, Any], year: int | None = None) -> pd.DataFrame:
    """Calculate coverage, load levels, required form and risk groups."""
    out = df.copy()
    if year is not None:
        out["analysis_year"] = year

    # Classify organization type based on org_type column value
    if "org_type" in out.columns:
        out["org_type_classified"] = out["org_type"].apply(classify_org_type)
    else:
        out["org_type_classified"] = "Другие"

    # Basic status.
    if "report_status" in out.columns:
        status = out["report_status"].map(clean_text).astype("string")
        out["is_report_filled"] = ~status.str.contains("не заполн", case=False, na=False)
    else:
        out["is_report_filled"] = True

    # Form level.
    form_levels = []
    for form_col, spec in scoring_config["forms"].items():
        flag = _safe_yes(out, form_col)
        level = int(spec["level"])
        out[f"{form_col}__yes"] = flag
        form_levels.append(flag * level)

    if form_levels:
        out["svk_form_level"] = pd.concat(form_levels, axis=1).max(axis=1).fillna(0).astype(int)
    else:
        out["svk_form_level"] = 0
    out["svk_form_name"] = out["svk_form_level"].map(_form_name)

    thresholds = scoring_config.get("load_level_thresholds", {})

    direction_keys: list[str] = []
    for key, spec in scoring_config["directions"].items():
        direction_keys.append(key)
        cover_col = spec["cover_column"]
        metrics = spec.get("metrics", [])

        out[f"{key}_covered"] = _safe_yes(out, cover_col).astype(int)

        metric_scores = []
        metric_values = []
        for metric in metrics:
            values = _safe_num(out, metric)
            out[f"{metric}__num"] = values
            metric_values.append(values)
            metric_scores.append(_percentile_scores(values))

        if metric_values:
            activity_sum = pd.concat(metric_values, axis=1).sum(axis=1)
            out[f"{key}_active"] = (activity_sum > 0).astype(int)
            out[f"{key}_load_score"] = pd.concat(metric_scores, axis=1).max(axis=1).fillna(0.0)
        else:
            out[f"{key}_active"] = 0
            out[f"{key}_load_score"] = 0.0

        out[f"{key}_load_level"] = _level_from_percentile(out[f"{key}_load_score"], thresholds)
        out[f"{key}_uncovered"] = ((out[f"{key}_active"] == 1) & (out[f"{key}_covered"] != 1)).astype(int)
        out[f"{key}_uncovered_load"] = out[f"{key}_uncovered"] * out[f"{key}_load_level"]

    active_cols = [f"{k}_active" for k in direction_keys]
    covered_cols = [f"{k}_covered" for k in direction_keys]
    level_cols = [f"{k}_load_level" for k in direction_keys]
    uncovered_cols = [f"{k}_uncovered" for k in direction_keys]
    uncovered_load_cols = [f"{k}_uncovered_load" for k in direction_keys]

    out["active_directions_count"] = out[active_cols].sum(axis=1)
    out["covered_active_directions_count"] = sum(
        ((out[f"{k}_active"] == 1) & (out[f"{k}_covered"] == 1)).astype(int) for k in direction_keys
    )
    out["coverage_share_active"] = np.where(
        out["active_directions_count"] > 0,
        out["covered_active_directions_count"] / out["active_directions_count"],
        np.nan,
    )
    out["uncovered_active_directions_count"] = out[uncovered_cols].sum(axis=1)
    out["uncovered_load_sum"] = out[uncovered_load_cols].sum(axis=1)
    out["max_uncovered_load"] = out[uncovered_load_cols].max(axis=1)
    out["max_direction_load_level"] = out[level_cols].max(axis=1)

    required = out["max_direction_load_level"].copy()
    bump_cfg = scoring_config.get("complexity_bump", {})
    if bump_cfg.get("enabled", True):
        min_count = int(bump_cfg.get("min_directions_count", 3))
        min_level = int(bump_cfg.get("min_direction_level", 2))
        bump_by = int(bump_cfg.get("bump_by", 1))
        complex_count = (out[level_cols] >= min_level).sum(axis=1)
        required = required + np.where(complex_count >= min_count, bump_by, 0)

    out["required_form_level"] = required.clip(lower=0, upper=4).astype(int)
    out["required_form_name"] = out["required_form_level"].map(_form_name)
    out["form_gap"] = out["svk_form_level"] - out["required_form_level"]

    out["risk_group"] = np.select(
        [
            ~out["is_report_filled"],
            (out["uncovered_active_directions_count"] > 0) & (out["form_gap"] < 0),
            out["uncovered_active_directions_count"] > 0,
            out["form_gap"] < 0,
        ],
        [
            "E. Отчет не заполнен / данные требуют проверки",
            "D. Двойной риск: непокрытие + слабая форма",
            "B. Недопокрытие активных направлений",
            "C. Слабая форма СВК при имеющейся нагрузке",
        ],
        default="A. Сбалансированная СВК",
    )

    out["contradictions"] = build_contradictions(out)
    out["anomalies"] = build_anomalies(out)
    out["data_quality_flags"] = build_data_quality_flags(out, direction_keys)
    return out


def build_contradictions(out: pd.DataFrame) -> pd.Series:
    """Build logical contradictions in data."""
    flags: list[list[str]] = [[] for _ in range(len(out))]

    def add(mask: pd.Series, text: str):
        for idx in out.index[mask.fillna(False)]:
            flags[out.index.get_loc(idx)].append(text)

    # 1. Устранено > выявленных нарушений
    if all(c in out.columns for c in ["violations_total", "violations_fixed_on_time", "violations_fixed_late"]):
        total = _safe_num(out, "violations_total")
        fixed = _safe_num(out, "violations_fixed_on_time") + _safe_num(out, "violations_fixed_late")
        add((total > 0) & (fixed > total), "Устраненных нарушений больше, чем выявленных")

    # 2. Отменено > принятых дисциплинарных решений
    if all(c in out.columns for c in ["disciplinary_decisions", "cancelled_decisions"]):
        decisions = _safe_num(out, "disciplinary_decisions")
        cancelled = _safe_num(out, "cancelled_decisions")
        add((decisions > 0) & (cancelled > decisions), "Отменено больше дисциплинарных решений, чем принято")

    # 3. Возвращено > направленных в правоохранительные органы
    if all(c in out.columns for c in ["materials_law_enforcement", "returned_refused"]):
        sent = _safe_num(out, "materials_law_enforcement")
        returned = _safe_num(out, "returned_refused")
        add((sent > 0) & (returned > sent), "Возвращено больше материалов, чем направлено в правоохранительные органы")

    # 4. Отказано > направленных в суд
    if all(c in out.columns for c in ["materials_court", "refused_satisfaction"]):
        sent = _safe_num(out, "materials_court")
        refused = _safe_num(out, "refused_satisfaction")
        add((sent > 0) & (refused > sent), "Отказано в удовлетворении больше материалов, чем направлено в суд")

    # 5. СВК не организован, но заполнены элементы СВК
    if "svk_organized" in out.columns:
        not_organized = _safe_yes(out, "svk_organized") != 1
        has_elements = pd.Series(False, index=out.index)
        for col in ["accounting_policy", "lna_approved", "powers_defined", "plan_schedule"]:
            if col in out.columns:
                has_elements = has_elements | (_safe_yes(out, col) == 1)
        add(not_organized & has_elements, "СВК не организован, но заполнены элементы СВК")

    return pd.Series(["; ".join(items) if items else "" for items in flags], index=out.index)


def build_anomalies(out: pd.DataFrame) -> pd.Series:
    """Build anomaly flags for suspicious data patterns."""
    flags: list[list[str]] = [[] for _ in range(len(out))]

    def add(mask: pd.Series, text: str):
        for idx in out.index[mask.fillna(False)]:
            flags[out.index.get_loc(idx)].append(text)

    # 1. Аномально высокие значения (99-й перцентиль)
    anomaly_fields = {
        "violations_total": ("нарушений", 605),
        "staff_avg": ("среднесписочной численности", 4238),
        "cash_payments": ("кассовых выплат", 20175413543),
        "procurement_count": ("закупок", 11941),
        "procurement_amount": ("объема закупок", 6138122952),
        "fkhz_count": ("ФХЖ", 2392993),
    }
    
    for field, (label, threshold) in anomaly_fields.items():
        if field in out.columns:
            values = _safe_num(out, field)
            add(values > threshold, f"Аномально высокое значение {label} (>{threshold:,.0f})")

    # 2. Одинаковые значения в 3+ полях (подозрение на копипаст)
    numeric_fields = ["violations_total", "violations_fixed_on_time", "violations_fixed_late", 
                      "remediation_plans", "lna_changes"]
    if all(col in out.columns for col in numeric_fields):
        for idx in out.index:
            values = []
            for col in numeric_fields:
                val = _safe_num(out, col).loc[idx]
                if val > 0:
                    values.append(val)
            if len(values) >= 3 and len(set(values)) == 1:
                flags[out.index.get_loc(idx)].append(f"Подозрительно одинаковые значения в нескольких полях ({values[0]})")

    # 3. Ноль нарушений при большой организации
    if all(c in out.columns for c in ["violations_total", "staff_avg"]):
        staff = _safe_num(out, "staff_avg")
        violations = _safe_num(out, "violations_total")
        staff_threshold = staff.quantile(0.75) if len(staff) > 0 else 0
        add((staff > staff_threshold) & (violations == 0), f"Ноль нарушений при большой организации (персонал >{staff_threshold:.0f})")

    # 4. Ноль закупок при большой организации
    if all(c in out.columns for c in ["procurement_count", "staff_avg"]):
        staff = _safe_num(out, "staff_avg")
        procurement = _safe_num(out, "procurement_count")
        staff_threshold = staff.quantile(0.75) if len(staff) > 0 else 0
        add((staff > staff_threshold) & (procurement == 0), f"Ноль закупок при большой организации (персонал >{staff_threshold:.0f})")

    # 5. Аномально высокое соотношение нарушений к ФХЖ
    if all(c in out.columns for c in ["violations_total", "fkhz_count"]):
        violations = _safe_num(out, "violations_total")
        fkhz = _safe_num(out, "fkhz_count")
        ratio = violations / fkhz.replace(0, np.nan)
        ratio_threshold = ratio.quantile(0.99) if len(ratio.dropna()) > 0 else 999
        add((ratio > ratio_threshold) & ratio.notna(), f"Аномально высокое соотношение нарушений к ФХЖ (>{ratio_threshold:.4f})")

    return pd.Series(["; ".join(items) if items else "" for items in flags], index=out.index)


def build_data_quality_flags(out: pd.DataFrame, direction_keys: list[str]) -> pd.Series:
    """Legacy function for backward compatibility. Combines contradictions, anomalies and direction coverage issues."""
    flags: list[list[str]] = [[] for _ in range(len(out))]

    def add(mask: pd.Series, text: str):
        for idx in out.index[mask.fillna(False)]:
            flags[out.index.get_loc(idx)].append(text)

    if "svk_organized" in out.columns:
        organized = _safe_yes(out, "svk_organized") == 1
        missing_basics = pd.Series(False, index=out.index)
        for col in ["accounting_policy", "lna_approved", "powers_defined", "plan_schedule"]:
            if col in out.columns:
                missing_basics = missing_basics | (_safe_yes(out, col) != 1)
        add(organized & missing_basics, "СВК организован, но отсутствует один или несколько базовых элементов")

    if all(c in out.columns for c in ["violations_total", "violations_fixed_on_time", "violations_fixed_late"]):
        total = _safe_num(out, "violations_total")
        fixed = _safe_num(out, "violations_fixed_on_time") + _safe_num(out, "violations_fixed_late")
        add((total > 0) & (fixed > total), "Устраненных нарушений больше, чем выявленных")

    add((out["max_direction_load_level"] >= 3) & (out["svk_form_level"] <= 1), "Высокая нагрузка при отсутствующей/минимальной форме СВК")

    for key in direction_keys:
        add((out[f"{key}_load_level"] >= 3) & (out[f"{key}_covered"] != 1), f"Высокая нагрузка по направлению '{key}', но направление не покрыто СВК")

    return pd.Series(["; ".join(items) if items else "" for items in flags], index=out.index)
