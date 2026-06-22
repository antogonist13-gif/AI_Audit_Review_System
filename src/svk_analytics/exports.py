from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .summaries import (
    anomalies_table,
    by_dimension_summary,
    contradictions_table,
    data_quality_issues,
    direction_summary,
    effectiveness_review_summary,
    improvement_actions_summary,
    normalized_activity_metrics,
    overview_metrics,
    report_status_summary,
    risk_group_summary,
    risk_methodology_summary,
    svk_elements_summary,
    svk_form_flags_summary,
    svk_form_level_summary,
    svk_full_basic_package_summary,
    top_risk_organizations,
    violations_summary,
)


def export_outputs(df: pd.DataFrame, scoring_config: dict[str, Any], output_dir: str | Path = "outputs") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    enriched_path = output_dir / "enriched_organizations.csv"
    df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    paths["enriched_csv"] = enriched_path

    direction_path = output_dir / "direction_summary.csv"
    direction_summary(df, scoring_config).to_csv(direction_path, index=False, encoding="utf-8-sig")
    paths["direction_summary_csv"] = direction_path

    risk_path = output_dir / "risk_groups.csv"
    risk_group_summary(df).to_csv(risk_path, index=False, encoding="utf-8-sig")
    paths["risk_groups_csv"] = risk_path

    general_path = output_dir / "general_statistics.csv"
    overview_metrics(df).to_csv(general_path, index=False, encoding="utf-8-sig")
    paths["general_statistics_csv"] = general_path

    quality_path = output_dir / "data_quality_issues.csv"
    data_quality_issues(df).to_csv(quality_path, index=False, encoding="utf-8-sig")
    paths["quality_issues_csv"] = quality_path

    contradictions_path = output_dir / "contradictions.csv"
    contradictions_table(df).to_csv(contradictions_path, index=False, encoding="utf-8-sig")
    paths["contradictions_csv"] = contradictions_path

    anomalies_path = output_dir / "anomalies.csv"
    anomalies_table(df).to_csv(anomalies_path, index=False, encoding="utf-8-sig")
    paths["anomalies_csv"] = anomalies_path

    xlsx_path = output_dir / "metrics_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Общая статистика из первого аналитического блока.
        overview_metrics(df).to_excel(writer, sheet_name="overview", index=False)
        report_status_summary(df).to_excel(writer, sheet_name="report_status", index=False)
        svk_elements_summary(df).to_excel(writer, sheet_name="svk_elements", index=False)
        svk_full_basic_package_summary(df).to_excel(writer, sheet_name="basic_package", index=False)
        svk_form_flags_summary(df, scoring_config).to_excel(writer, sheet_name="svk_form_flags", index=False)
        svk_form_level_summary(df).to_excel(writer, sheet_name="svk_form_level", index=False)
        risk_methodology_summary(df).to_excel(writer, sheet_name="risk_methodology", index=False)
        effectiveness_review_summary(df).to_excel(writer, sheet_name="effectiveness_review", index=False)
        improvement_actions_summary(df).to_excel(writer, sheet_name="improvement_actions", index=False)
        violations_summary(df).to_excel(writer, sheet_name="violations", index=False)
        normalized_activity_metrics(df).to_excel(writer, sheet_name="normalized_metrics", index=False)

        # Блок соразмерности формы СВК и направлений деятельности.
        direction_summary(df, scoring_config).to_excel(writer, sheet_name="directions", index=False)
        risk_group_summary(df).to_excel(writer, sheet_name="risk_groups", index=False)
        top_risk_organizations(df, limit=100).to_excel(writer, sheet_name="top_risk_orgs", index=False)
        contradictions_table(df).to_excel(writer, sheet_name="contradictions", index=False)
        anomalies_table(df).to_excel(writer, sheet_name="anomalies", index=False)
        data_quality_issues(df).to_excel(writer, sheet_name="quality_issues", index=False)
        by_dimension_summary(df, "federal_district").to_excel(writer, sheet_name="by_federal_district", index=False)
        by_dimension_summary(df, "org_type").to_excel(writer, sheet_name="by_org_type", index=False)
    paths["summary_xlsx"] = xlsx_path

    return paths
