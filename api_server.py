"""
API endpoints для Next.js frontend
Этот модуль предоставляет REST API для получения данных аналитики СВК
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
import pandas as pd

from src.svk_analytics.columns import build_canonical_frame, resolve_columns
from src.svk_analytics.io import find_latest_raw_file, load_report, load_yaml
from src.svk_analytics.scoring import enrich_with_metrics
from src.svk_analytics.summaries import (
    overview_metrics,
    violations_and_remediation,
    management_actions,
    legal_responsibility_measures,
    normalized_activity_metrics,
    svk_elements_summary,
    svk_form_level_summary,
    svk_form_flags_summary,
    risk_methodology_summary,
    effectiveness_review_summary,
    improvement_actions_summary,
    direction_summary,
    by_dimension_summary,
    top_risk_organizations,
    contradictions_table,
    anomalies_table,
    report_status_summary,
)

app = FastAPI(title="СВК Analytics API", version="1.0.0")

# CORS для Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное хранилище данных
_cached_data = {}


def load_data(year: int = 2025):
    """Загружает и кеширует данные"""
    cache_key = f"data_{year}"
    
    if cache_key in _cached_data:
        return _cached_data[cache_key]
    
    columns_config = load_yaml("config/columns.yml")
    scoring_config = load_yaml("config/scoring.yml")
    
    latest = find_latest_raw_file("data/raw")
    if not latest:
        raise HTTPException(status_code=404, detail="No data file found")
    
    raw = load_report(str(latest))
    resolved, resolution_report = resolve_columns(raw, columns_config)
    canonical = build_canonical_frame(raw, resolved)
    enriched = enrich_with_metrics(canonical, scoring_config, year=year)
    
    _cached_data[cache_key] = {
        "df": enriched,
        "scoring_config": scoring_config,
        "resolution_report": resolution_report,
    }
    
    return _cached_data[cache_key]


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Применяет фильтры к датафрейму"""
    filtered = df.copy()
    
    if filters.get("federal_district"):
        filtered = filtered[filtered["federal_district"].isin(filters["federal_district"])]
    if filters.get("region"):
        filtered = filtered[filtered["region"].isin(filters["region"])]
    if filters.get("org_type"):
        filtered = filtered[filtered["org_type_classified"].isin(filters["org_type"])]
    if filters.get("risk_group"):
        filtered = filtered[filtered["risk_group"].isin(filters["risk_group"])]
    
    return filtered


@app.get("/api/overview")
async def get_overview(
    year: int = Query(2025),
    federal_district: Optional[List[str]] = Query(None),
    region: Optional[List[str]] = Query(None),
    org_type: Optional[List[str]] = Query(None),
    risk_group: Optional[List[str]] = Query(None),
):
    """Получить обзорные метрики"""
    data = load_data(year)
    df = data["df"]
    
    filters = {
        "federal_district": federal_district,
        "region": region,
        "org_type": org_type,
        "risk_group": risk_group,
    }
    filtered = apply_filters(df, filters)
    
    ov = overview_metrics(filtered)
    
    # Дополнительные метрики
    avg_coverage = filtered["coverage_share_active"].mean()
    uncovered_count = int((filtered["uncovered_active_directions_count"] > 0).sum())
    weak_form_count = int((filtered["form_gap"] < 0).sum())
    double_risk_count = int(filtered["risk_group"].str.startswith("D.").sum())
    
    return {
        "total_organizations": len(filtered),
        "overview_metrics": ov.to_dict(orient="records"),
        "additional_metrics": {
            "avg_coverage": float(avg_coverage * 100) if not pd.isna(avg_coverage) else None,
            "uncovered_count": uncovered_count,
            "weak_form_count": weak_form_count,
            "double_risk_count": double_risk_count,
        }
    }


@app.get("/api/violations")
async def get_violations(year: int = Query(2025)):
    """Получить данные о нарушениях"""
    data = load_data(year)
    df = data["df"]
    
    violations = violations_and_remediation(df)
    mgmt = management_actions(df)
    legal = legal_responsibility_measures(df)
    
    return {
        "violations_remediation": violations.to_dict(orient="records"),
        "management_actions": mgmt.to_dict(orient="records"),
        "legal_measures": legal.to_dict(orient="records"),
    }


@app.get("/api/maturity")
async def get_maturity(year: int = Query(2025)):
    """Получить данные о зрелости СВК"""
    data = load_data(year)
    df = data["df"]
    scoring_config = data["scoring_config"]
    
    elements = svk_elements_summary(df)
    form_level = svk_form_level_summary(df)
    form_flags = svk_form_flags_summary(df, scoring_config)
    risk_method = risk_methodology_summary(df)
    effectiveness = effectiveness_review_summary(df)
    improvement = improvement_actions_summary(df)
    
    return {
        "elements": elements.to_dict(orient="records"),
        "form_level": form_level.to_dict(orient="records"),
        "form_flags": form_flags.to_dict(orient="records"),
        "risk_methodology": risk_method.to_dict(orient="records"),
        "effectiveness": effectiveness.to_dict(orient="records"),
        "improvement": improvement.to_dict(orient="records"),
    }


@app.get("/api/directions")
async def get_directions(year: int = Query(2025)):
    """Получить данные о покрытии направлений"""
    data = load_data(year)
    df = data["df"]
    scoring_config = data["scoring_config"]
    
    directions = direction_summary(df, scoring_config)
    
    return {
        "directions": directions.to_dict(orient="records")
    }


@app.get("/api/form-gap")
async def get_form_gap(year: int = Query(2025)):
    """Получить данные о соразмерности формы СВК (peer + рекомендуемая форма)"""
    data = load_data(year)
    df = data["df"]
    
    gap_counts = df.groupby("form_gap", dropna=False).size().reset_index(name="orgs")
    
    scatter_data = df[[
        "org_name", "svk_form_level", "recommended_form_level",
        "max_direction_load_level", "coverage_share_active", "risk_group"
    ]].to_dict(orient="records")

    peer_cols = [
        "org_name", "svk_form_level", "peer_form_median", "form_vs_peer",
        "peer_group_size", "peer_group_level", "risk_group",
    ]
    peer_scatter = df[[c for c in peer_cols if c in df.columns]].to_dict(orient="records")

    form_peer_counts = (
        df.groupby("form_vs_peer", dropna=False).size().reset_index(name="orgs")
        if "form_vs_peer" in df.columns else pd.DataFrame()
    )
    
    by_district = by_dimension_summary(df, "federal_district")
    
    return {
        "peer_distribution": form_peer_counts.to_dict(orient="records") if not form_peer_counts.empty else [],
        "peer_scatter_data": peer_scatter,
        "gap_distribution": gap_counts.to_dict(orient="records"),
        "scatter_data": scatter_data,
        "by_district": by_district.to_dict(orient="records"),
    }


@app.get("/api/risk-organizations")
async def get_risk_organizations(
    year: int = Query(2025),
    limit: int = Query(100)
):
    """Получить топ организаций в зоне риска"""
    data = load_data(year)
    df = data["df"]
    
    top = top_risk_organizations(df, limit=limit)
    
    return {
        "organizations": top.to_dict(orient="records")
    }


@app.get("/api/data-quality")
async def get_data_quality(year: int = Query(2025)):
    """Получить данные о качестве данных"""
    data = load_data(year)
    df = data["df"]
    
    contradictions = contradictions_table(df)
    anomalies = anomalies_table(df)
    status = report_status_summary(df)
    
    return {
        "contradictions": contradictions.to_dict(orient="records") if not contradictions.empty else [],
        "anomalies": anomalies.to_dict(orient="records") if not anomalies.empty else [],
        "report_status": status.to_dict(orient="records"),
    }


@app.get("/api/filters/options")
async def get_filter_options(year: int = Query(2025)):
    """Получить варианты для фильтров"""
    data = load_data(year)
    df = data["df"]
    
    return {
        "federal_districts": sorted(df["federal_district"].dropna().unique().tolist()),
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "org_types": sorted(df["org_type_classified"].dropna().unique().tolist()),
        "risk_groups": sorted(df["risk_group"].dropna().unique().tolist()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
