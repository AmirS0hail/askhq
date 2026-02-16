"""Quick connectivity check for Ollama.

Usage:
    python -m scripts.test_ollama
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    url = settings.ollama_url
    model = settings.ollama_model

    print(f"Checking Ollama at {url}...")

    # Check connectivity
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5.0)
        r.raise_for_status()
    except httpx.ConnectError:
        print(f"❌ Cannot connect to Ollama at {url}")
        print("   Start it with: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

    print("✅ Ollama is running")

    # Check model availability
    models = r.json().get("models", [])
    model_names = [m.get("name", "") for m in models]

    if any(model in name for name in model_names):
        print(f"✅ Model '{model}' is available")
    else:
        print(f"⚠️  Model '{model}' not found")
        print(f"   Available: {', '.join(model_names) or 'none'}")
        print(f"   Pull it with: ollama pull {model}")
        sys.exit(1)

    # Quick generate test
    print(f"\nTesting generation with {model}...")
    try:
        r = httpx.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": "Say 'hello' in one word.", "stream": False},
            timeout=30.0,
        )
        r.raise_for_status()
        resp = r.json().get("response", "").strip()
        print(f"✅ Model response: {resp[:100]}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)

    print("\n✅ Ollama is fully operational")


if __name__ == "__main__":
    main()
