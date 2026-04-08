# Project Context — AI Audit Review System

## Entry point
act_pipeline.py → analyze_act()

## Pipeline order (strict sequence)
1. act_preprocessor   → extracts Violation objects
2. violation_normalizer → lemmatization, keywords
3. act_retrieval       → hybrid search (vector + BM25)
4. violation_evaluator → 3x LLM scores (evidence, legal, actionability)
5. formulation_improver → rewrites violation text
6. checklist_builder   → final ChecklistItem with statuses

## Critical thresholds (config.py)
- EVIDENCE_SUFFICIENT_THRESHOLD = 0.4
- LEGAL_CORRECT_THRESHOLD = 0.5
- ACTIONABILITY_THRESHOLD = 0.5
- LAW_GROUNDING_FUZZY_MIN = 0.7

## Known problems (priority order)
1. Empty ChromaDB → retrieval returns nothing → LLM scores are blind
2. LLM (gemma3:12b) poorly follows structured prompts
3. _find_context_in_descriptive misses short law_ref
4. pipeline_stats is global (not thread-safe)

## Data flow
Violation → ViolationContext → ChecklistItem
           ↑ retrieval fills this

## LLM interface
rag_pipeline.py wraps Ollama with caching (llm_cache.py, TTL 24h)
All prompts are in prompts.py