from __future__ import annotations

import argparse
from pathlib import Path

from src.svk_analytics.columns import build_canonical_frame, resolve_columns
from src.svk_analytics.exports import export_outputs
from src.svk_analytics.io import load_report, load_yaml
from src.svk_analytics.scoring import enrich_with_metrics
from src.svk_analytics.summaries import direction_summary, overview_metrics, risk_group_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Расчет аналитики СВК по годовому сводному отчету")
    parser.add_argument("--input", required=True, help="Путь к исходному отчету")
    parser.add_argument("--year", type=int, default=None, help="Год анализа, например 2025")
    parser.add_argument("--columns-config", default="config/columns.yml")
    parser.add_argument("--scoring-config", default="config/scoring.yml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    columns_config = load_yaml(args.columns_config)
    scoring_config = load_yaml(args.scoring_config)

    raw = load_report(args.input)
    resolved, resolution_report = resolve_columns(raw, columns_config)

    missing = resolution_report.loc[~resolution_report["found"], ["canonical", "label"]]
    if not missing.empty:
        print("\nВнимание: часть колонок не найдена. Проверьте config/columns.yml")
        print(missing.to_string(index=False))

    canonical = build_canonical_frame(raw, resolved)
    enriched = enrich_with_metrics(canonical, scoring_config, year=args.year)

    paths = export_outputs(enriched, scoring_config, args.output_dir)

    print("\nГотово. Основные показатели:")
    print(overview_metrics(enriched).to_string(index=False))

    print("\nПокрытие направлений:")
    print(direction_summary(enriched, scoring_config).to_string(index=False))

    print("\nГруппы риска:")
    print(risk_group_summary(enriched).to_string(index=False))

    print("\nФайлы сохранены:")
    for name, path in paths.items():
        print(f"- {name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
