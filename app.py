"""Streamlit application — AI-assisted Audit Review System v2."""
from __future__ import annotations

import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import streamlit as st


@st.cache_resource
def _get_store() -> dict:
    """Singleton dict that survives Streamlit script reloads.

    Background threads write results here; the main Streamlit thread reads and
    copies values into session_state (the only place UI state may be mutated).
    """
    return {}


logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ИИ-система аудиторской проверки",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .status-sufficient { color: #28a745; font-weight: bold; }
    .status-unclear { color: #fd7e14; font-weight: bold; }
    .status-insufficient { color: #dc3545; font-weight: bold; }
    .status-unknown { color: #6c757d; }
    .warning-badge {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Results renderer (defined before use)
# ---------------------------------------------------------------------------

def _render_results(items, act_name, pipeline_stats):
    """Render the full results table and download button."""
    st.subheader(f"📋 Результаты — нарушений: {len(items)}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    possibly_weak = sum(1 for i in items if i.possibly_not_a_violation)
    ungrounded = sum(1 for i in items if not i.legal_qualification_grounded)
    avg_conf = sum(i.confidence_score for i in items) / len(items) if items else 0

    col1.metric("Всего нарушений", len(items))
    col2.metric("Средняя уверенность", f"{avg_conf:.0%}")
    col3.metric("⚠️ Возможно не нарушения", possibly_weak)
    col4.metric("⚠️ Необоснованные нормы", ungrounded)

    # Filters
    st.markdown("**Фильтры:**")
    f1, f2 = st.columns(2)
    show_weak = f1.checkbox("Только возможно не нарушения", value=False)
    show_ungrounded = f2.checkbox("Только необоснованные нормы", value=False)

    filtered = items
    if show_weak:
        filtered = [i for i in filtered if i.possibly_not_a_violation]
    if show_ungrounded:
        filtered = [i for i in filtered if not i.legal_qualification_grounded]

    # Detailed item cards
    for idx, item in enumerate(filtered, start=1):
        flags = []
        if item.possibly_not_a_violation:
            flags.append("⚠️ Возможно не нарушение")
        if not item.legal_qualification_grounded:
            flags.append("⚠️ Необоснованная норма")

        flag_str = " | ".join(flags)
        title_text = item.raw_text[:120] + ("…" if len(item.raw_text) > 120 else "")
        with st.expander(
            f"**#{idx}** — {title_text} | Уверенность: {item.confidence_score:.0%} {flag_str}",
            expanded=(idx == 1),
        ):
            tab1, tab2, tab3 = st.tabs(["📝 Детали", "📊 Оценки", "🔍 Трассировка"])

            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Источник:** {item.source_document} стр.{item.page}")
                    st.markdown(f"**Раздел:** {item.section}")
                    st.markdown(f"**Субъект:** {item.subject or '—'}")
                    st.markdown(f"**Ссылка на НПА:** `{item.law_ref or '—'}`")
                    st.markdown("**Формулировка (таблица):**")
                    st.text(item.raw_text)
                    st.markdown("**Контекст (описательная часть):**")
                    st.text(item.description[:600])
                with col_b:
                    st.markdown("**Улучшенная формулировка:**")
                    st.text(item.improved_formulation[:300] if item.improved_formulation else "—")
                    st.markdown(
                        f"**Правовая квалификация:** {item.legal_qualification or '—'} "
                        f"{'✅' if item.legal_qualification_grounded else '⚠️ Не подтверждено'}"
                    )
                    if item.recommendation:
                        st.markdown("**Рекомендация:**")
                        st.info(item.recommendation[:300])
                    if item.evidence_comment:
                        st.markdown("**Комментарий к доказательствам:**")
                        st.caption(item.evidence_comment[:200])

            with tab2:
                score_data = {
                    "Критерий": ["Доказательства", "Законность", "Исполнимость", "Уверенность"],
                    "Оценка": [
                        item.evidence_score or 0,
                        item.legal_score or 0,
                        item.actionability_score or 0,
                        item.confidence_score,
                    ],
                    "Статус": [
                        item.evidence_status,
                        item.legal_status,
                        item.actionability_status,
                        "—",
                    ],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

                if item.verification_notes:
                    st.markdown("**Заметки верификатора:**")
                    for note in item.verification_notes:
                        icon = "✅" if note.status == "confirmed" else "⚠️"
                        st.caption(f"{icon} {note.axis}: {note.status} — {note.detail}")

            with tab3:
                if item.trace:
                    st.markdown(f"**ID нарушения:** `{item.violation_id}`")
                    st.markdown(f"**Использовано фрагментов:** {len(item.trace.used_chunk_ids)}")
                    if item.trace.used_chunk_ids:
                        st.code("\n".join(item.trace.used_chunk_ids[:10]))
                    st.markdown("**Запросы к базе знаний:**")
                    for k, q in item.trace.retrieval_queries.items():
                        st.caption(f"_{k}_: {q[:80]}")
                else:
                    st.info("Данные трассировки отсутствуют.")

    # Download report
    st.divider()
    st.subheader("📥 Скачать отчёт")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if st.button("Сформировать отчёт .docx"):
            try:
                from report_generator import generate_report_bytes
                st.session_state.report_bytes = generate_report_bytes(
                    items, act_name=act_name, pipeline_stats=pipeline_stats
                )
                st.success("Отчёт готов. Нажмите кнопку ниже для скачивания.")
            except Exception as exc:
                st.error(f"Ошибка формирования отчёта: {exc}")
        if st.session_state.get("report_bytes"):
            st.download_button(
                "⬇️ Скачать отчёт (.docx)",
                data=st.session_state.report_bytes,
                file_name=f"audit_report_{act_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    with col_d2:
        if st.button("Сформировать Markdown-отчёт"):
            try:
                import tempfile
                from report_generator import _generate_markdown
                with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                    _generate_markdown(items, tmp.name, act_name=act_name, pipeline_stats=pipeline_stats)
                    md_text = Path(tmp.name).read_text(encoding="utf-8")
                st.download_button(
                    "⬇️ Скачать Markdown",
                    data=md_text,
                    file_name=f"audit_report_{act_name}.md",
                    mime="text/markdown",
                )
            except Exception as exc:
                st.error(f"Ошибка формирования отчёта: {exc}")


# ---------------------------------------------------------------------------
# Sidebar — settings and pipeline stats
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ ИИ-аудит")
    st.markdown("**v2.0** — RAG-анализ нарушений")
    st.divider()

    st.subheader("⚙️ Настройки")
    ollama_model = st.selectbox(
        "Модель LLM",
        ["gemma3:12b", "llama3:8b", "mistral:7b", "qwen2.5:7b"],
        index=0,
    )
    max_workers = st.slider("Параллельные потоки", 1, 8, 4)
    index_act = st.checkbox("Индексировать акт в справочную БД", value=False)

    st.divider()
    st.subheader("📊 Статистика пайплайна")
    stats_placeholder = st.empty()

    with stats_placeholder.container():
        st.info("Запустите анализ для просмотра статистики")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Система аудиторской проверки на базе ИИ")
st.markdown(
    "Загрузите аудиторский акт (PDF или DOCX) для автоматического извлечения, "
    "оценки и верификации нарушений с помощью RAG + LLM анализа."
)

# Reference document indexing expander
with st.expander("📚 Индексировать справочные документы", expanded=False):
    st.markdown(
        "Загрузите нормативные документы, каталоги типовых нарушений "
        "или исторические чек-листы для пополнения справочной базы данных."
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Нормативные документы**")
        norm_files = st.file_uploader(
            "Загрузить нормы (PDF/DOCX)",
            accept_multiple_files=True,
            key="norm_upload",
        )
        if norm_files and st.button("Индексировать нормы"):
            with st.spinner("Индексирование нормативных документов..."):
                try:
                    from multi_indexer import index_norms
                    import tempfile, os
                    paths = []
                    for f in norm_files:
                        tmp = tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(f.name).suffix
                        )
                        tmp.write(f.read())
                        tmp.close()
                        paths.append(tmp.name)
                    index_norms(paths)
                    st.success(f"Проиндексировано нормативных документов: {len(paths)}")
                except Exception as exc:
                    st.error(f"Ошибка индексации: {exc}")

    with col2:
        st.markdown("**Типовые нарушения**")
        typical_text = st.text_area(
            "JSON-список нарушений",
            placeholder='[{"text": "нарушение...", "law_ref": "ФЗ-294"}]',
            height=100,
            key="typical_input",
        )
        if typical_text and st.button("Индексировать типовые нарушения"):
            try:
                import json
                items = json.loads(typical_text)
                from multi_indexer import index_typical_violations
                index_typical_violations(items)
                st.success(f"Проиндексировано типовых нарушений: {len(items)}")
            except Exception as exc:
                st.error(f"Ошибка: {exc}")

    with col3:
        st.markdown("**Исторические чек-листы**")
        hist_files = st.file_uploader(
            "Загрузить чек-листы (PDF/DOCX)",
            accept_multiple_files=True,
            key="hist_upload",
        )
        if hist_files and st.button("Индексировать исторические"):
            with st.spinner("Индексирование исторических чек-листов..."):
                try:
                    from multi_indexer import index_historical_checklists
                    paths = []
                    for f in hist_files:
                        tmp = tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(f.name).suffix
                        )
                        tmp.write(f.read())
                        tmp.close()
                        paths.append(tmp.name)
                    index_historical_checklists(paths)
                    st.success(f"Проиндексировано исторических документов: {len(paths)}")
                except Exception as exc:
                    st.error(f"Ошибка индексации: {exc}")

st.divider()

# Act upload + analysis
uploaded_file = st.file_uploader(
    "📄 Загрузить аудиторский акт (PDF или DOCX)",
    type=["pdf", "docx", "txt"],
    label_visibility="visible",
)

if uploaded_file is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"Файл готов: **{uploaded_file.name}** ({uploaded_file.size:,} байт)")
    with col_btn:
        analyze_btn = st.button("🔍 Анализировать акт", type="primary", use_container_width=True)

    if analyze_btn:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Capture sidebar values before thread starts
        _ollama_model = ollama_model
        _index_act = index_act
        _max_workers = max_workers
        _act_name = uploaded_file.name

        # Unique key per session so multiple tabs don't collide
        _skey = st.session_state.get("_session_key") or str(uuid.uuid4())
        st.session_state["_session_key"] = _skey

        # Reset session state (main thread — safe)
        st.session_state["_analysis_status"] = "running"
        st.session_state["_analysis_error"] = None
        st.session_state["items"] = None
        st.session_state["pipeline_stats"] = None
        st.session_state["act_name"] = _act_name
        st.session_state["report_bytes"] = None

        # Reset store for this session (using the cache_resource singleton)
        _get_store()[_skey] = {"status": "running", "items": None, "stats": None, "error": None}

        def _run_analysis(_sk=_skey, _tmp=tmp_path, _model=_ollama_model,
                          _ia=_index_act, _mw=_max_workers):
            _store = _get_store()
            try:
                import config as _cfg
                _cfg.OLLAMA_MODEL = _model
                from act_pipeline import analyze_act
                result = analyze_act(_tmp, index_act=_ia, max_workers=_mw)
                from act_pipeline import pipeline_stats as _ps
                _store[_sk]["items"] = result
                _store[_sk]["stats"] = dict(_ps)
                _store[_sk]["status"] = "done"
            except Exception as exc:
                _store[_sk]["error"] = str(exc)
                _store[_sk]["status"] = "error"
                logging.exception("Analysis error in background thread")

        threading.Thread(target=_run_analysis, daemon=True).start()
        st.rerun()

# ---------------------------------------------------------------------------
# Analysis status & results (outside file-upload block so reruns work)
# ---------------------------------------------------------------------------

_status = st.session_state.get("_analysis_status")

if _status == "running":
    _skey = st.session_state.get("_session_key", "")
    _store = _get_store().get(_skey, {})
    _store_status = _store.get("status", "running")

    if _store_status == "done":
        st.session_state["items"] = _store["items"]
        st.session_state["pipeline_stats"] = _store["stats"]
        st.session_state["_analysis_status"] = "done"
        st.rerun()
    elif _store_status == "error":
        st.session_state["_analysis_error"] = _store["error"]
        st.session_state["_analysis_status"] = "error"
        st.rerun()
    else:
        st.info("⏳ Анализ выполняется в фоне. Страница обновится автоматически...")
        with st.spinner("Анализируем нарушения..."):
            time.sleep(3)
        st.rerun()

elif _status == "error":
    st.error(f"Ошибка анализа: {st.session_state.get('_analysis_error', 'Неизвестная ошибка')}")

elif _status == "done" and st.session_state.get("items"):
    _items = st.session_state["items"]
    _ps = st.session_state.get("pipeline_stats") or {}
    _act_name = st.session_state.get("act_name", "")

    # Update sidebar stats
    with stats_placeholder.container():
        st.metric("Всего нарушений", _ps.get("violations_total", "—"))
        st.metric("Обработано", _ps.get("violations_processed", "—"))
        st.metric("Ошибок", _ps.get("violations_failed", "—"))
        st.metric("Запросов к БД", _ps.get("retrieval_calls", "—"))
        st.metric(
            "Средняя уверенность",
            f"{_ps['avg_confidence']:.2f}" if isinstance(_ps.get("avg_confidence"), float) else "—",
        )
        st.metric("Попаданий в кэш LLM", _ps.get("llm_cache_hits", "—"))
        st.metric("Сбоев обоснования", _ps.get("grounding_failures", "—"))
        st.metric("Переопределений верификатора", _ps.get("verifier_overrides", "—"))

    st.success(f"✅ Анализ завершён — обработано нарушений: {len(_items)}")
    if not _items:
        st.warning("В документе нарушений не обнаружено.")
    else:
        _render_results(_items, _act_name, _ps)

elif st.session_state.get("items"):
    _render_results(
        st.session_state["items"],
        st.session_state.get("act_name", ""),
        st.session_state.get("pipeline_stats") or {},
    )
