import re
from typing import Any, List

def normalize_answer(ans: Any) -> List[str]:
    """Normalize ground-truth answers to a list of lowercase strings."""
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans.strip().lower()]
    if isinstance(ans, (list, tuple)):
        out: List[str] = []
        for a in ans:
            if a is None:
                continue
            out.append(str(a).strip().lower())
        return out
    return [str(ans).strip().lower()]

def parse_assistant_response(raw: str) -> str:
    """
    Parse answer from the model's response.

    Strategy:
      1. Find the last occurrence of 'assistant:' (lowercase to be safe)
      2. Take everything after it until end.
      3. Remove any leading whitespace/newlines.
      4. Strip common trailing stop punctuation or residual prompt echoes.
      5. Collapse internal whitespace.
    """
    lower_raw = raw.lower()
    key = 'assistant:'
    pos = lower_raw.rfind(key)
    answer_fragment = raw.strip()
    if pos != -1:
        answer_fragment = raw[pos + len(key):].strip()

    # Remove a leading comma or period if model added punctuation right after colon
    answer_fragment = re.sub(r'^[.,:;!?\s]+', '', answer_fragment)
    # If the model echoed the entire instruction again, split on common delimiters.
    # Heuristic: take first sentence/clause before double newline or line break.
    answer_fragment = answer_fragment.split('\n')[0].strip()
    # Remove trailing artifacts like stray prompt words
    # Example pattern: "green.," -> "green"
    answer_fragment = re.sub(r'[.,:;!?]+$', '', answer_fragment)
    # Final normalization
    answer_fragment = answer_fragment.strip().lower()
    answer_fragment = re.sub(r'%', '', answer_fragment)
    return answer_fragment
