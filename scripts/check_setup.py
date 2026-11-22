"""Stage 0 smoke test: is the environment ready to talk to the LLM?

Run from the project root:
    python scripts/check_setup.py
"""

import sys
from pathlib import Path

# Let this script import the `reviewlens` package from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewlens.config import ConfigError, get_llm, get_model_name  # noqa: E402


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Model   : {get_model_name()}")

    try:
        llm = get_llm()
    except ConfigError as err:
        print(f"[FAIL] {err}")
        return 1
    print("API key : found in environment (value hidden)")

    try:
        reply = llm.invoke("Reply with exactly: ReviewLens is ready.")
    except Exception as err:  # show the real reason: bad key, unknown model, no credit...
        print(f"[FAIL] LLM call failed: {type(err).__name__}: {err}")
        return 1

    print(f"LLM says: {reply.content}")
    usage = reply.usage_metadata or {}
    print(f"Tokens  : {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out")
    print("[OK] Setup works. Ready for Stage 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
