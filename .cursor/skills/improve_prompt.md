# Skill: improve_prompt

## Trigger
Use this skill when an existing prompt in prompts.py produces unreliable output
from a local LLM (Ollama / gemma3 / mistral).

## Input required from user
- PROMPT_NAME: which prompt to improve (e.g. "evidence evaluator")
- CURRENT_PROMPT: paste the current prompt text
- FAILURE_EXAMPLE: paste 1-2 examples of bad LLM output

## Execution steps (follow in order)

### Step 1 — Diagnose
Identify which of these failure modes is happening:
- [ ] LLM returns prose instead of JSON
- [ ] LLM returns JSON but wrong field names
- [ ] Score is always 0.0 or 1.0 (no gradient)
- [ ] Parser crashes (KeyError, ValueError, AttributeError)
- [ ] LLM adds markdown fences (```json ... ```)

### Step 2 — Rewrite prompt
Apply all of these fixes:
1. Add explicit role: "Ты — аудитор, оценивающий [конкретный аспект]"
2. Add strict output instruction at TOP and BOTTOM of prompt:
   "Ответь ТОЛЬКО JSON. Без пояснений. Без markdown."
3. Add labeled input block:
   "### НАРУШЕНИЕ:\n{violation_text}"
4. Add explicit JSON schema:
   "Формат ответа: {\"score\": float 0.0-1.0, \"reasoning\": \"строка до 2 предложений\"}"
5. Add 1 positive + 1 negative few-shot example

### Step 3 — Rewrite parser
Rewrite the paired parse_ function with:
```python
def parse_PROMPT_NAME(raw: str) -> dict:
    import json, logging
    logging.debug(f"[parse_PROMPT_NAME] raw response: {raw!r}")
    try:
        # Strip markdown fences if present
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(clean)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # clamp
        reasoning = str(data.get("reasoning", ""))
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        logging.warning(f"[parse_PROMPT_NAME] parse failed: {e}, raw: {raw!r}")
        return {"score": 0.0, "reasoning": "parse_error"}
```

### Step 4 — Deliver
Output:
1. BEFORE prompt (summarized)
2. AFTER prompt (full)
3. AFTER parse_ function (full)
4. Two test cases to verify manually