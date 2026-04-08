# Skill: debug_pipeline

## Trigger
Use this skill when a pipeline step produces wrong output, crashes,
or a violation silently disappears from the checklist.

## Input required from user
- SYMPTOM: what's wrong (crash / wrong score / missing item / freeze)
- ERROR: paste traceback if available
- VIOLATION: paste the violation text that triggered the problem

## Execution steps (follow in order)

### Step 1 — Locate the step
Map symptom to pipeline step:
| Symptom | Likely step |
|---|---|
| Violation not extracted | act_preprocessor |
| Keywords missing / wrong | violation_normalizer |
| Score always 0.0 | act_retrieval (empty results) |
| Score None / parse error | violation_evaluator + prompts.py |
| Grounding always False | formulation_improver (empty norms) |
| Status wrong | checklist_builder + verifier |
| Streamlit freeze | pipeline_stats thread issue |

### Step 2 — Add targeted logging
Insert this at the suspected step entry and exit:
```python
import logging
logging.debug(f"[STEP_NAME] INPUT: {input_object!r}")
# ... step logic ...
logging.debug(f"[STEP_NAME] OUTPUT: {output_object!r}")
```

### Step 3 — Check config thresholds
Before assuming code bug, verify:
- Is score below threshold? → check config.py values
- Is this a known limitation? → check PROJECT_CONTEXT.md "Known problems"

### Step 4 — Isolate with minimal reproducer
Write a 10-line script to reproduce outside Streamlit:
```python
from models import Violation
from act_pipeline import analyze_act

v = Violation(text="[paste violation here]", law_ref="", source="test")
result = analyze_act([v])
print(result)
```

### Step 5 — Deliver fix
Output:
1. Root cause (one sentence)
2. Affected file + function + line
3. Before/after code change (minimal)
4. How to verify the fix worked