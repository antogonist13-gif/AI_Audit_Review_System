# Skill: add_tests

## Trigger
Use this skill when you need to write tests for a pipeline function,
a prompt parser, or a new module.

## Input required from user
- TARGET: file and function name to test (e.g. "prompts.py → parse_evidence")
- BEHAVIOUR: what it should do (1-2 sentences)
- EDGE_CASES: any known failure modes to cover

## Execution steps (follow in order)

### Step 1 — Classify test type
| Target | Test type | Mock needed |
|---|---|---|
| parse_ functions in prompts.py | Unit | No |
| violation_evaluator | Unit | Mock Ollama |
| act_retrieval | Unit | Mock ChromaDB + BM25 |
| act_pipeline.analyze_act | Integration | Mock all external |
| checklist_builder / verifier | Unit | No |

### Step 2 — Apply standard mock pattern
```python
from unittest.mock import patch, MagicMock
import pytest

# For Ollama mock:
@patch("rag_pipeline.OllamaClient")
def test_name(mock_ollama):
    mock_ollama.return_value.generate.return_value = (
        '{"score": 0.75, "reasoning": "test reasoning"}'
    )

# For ChromaDB mock:
@patch("act_retrieval.ChromaDBClient")
def test_name(mock_chroma):
    mock_chroma.return_value.query.return_value = {"documents": [], "distances": []}
```

### Step 3 — Always cover these 4 cases
For every function, write tests for:
1. **Happy path** — valid input, expected output
2. **Empty input** — empty string / empty list / None
3. **Boundary value** — score exactly at threshold (from config.py)
4. **Malformed LLM output** — unparseable string, wrong JSON fields

### Step 4 — Deliver
Output:
1. Test file path (tests/test_FILENAME.py)
2. Full test code (ready to run with pytest)
3. Command to run only these tests:
   pytest tests/test_FILENAME.py -v