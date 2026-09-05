from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.gradio_app import create_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gradio UI for RAG demo")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode")
    args = parser.parse_args()

    demo = create_demo(api_base_url=args.api_base_url)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
