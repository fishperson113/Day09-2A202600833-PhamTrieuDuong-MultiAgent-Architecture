from __future__ import annotations

import argparse
from pathlib import Path

from app.graph import ShoppingAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shopping Assistant CLI")
    parser.add_argument("--question", help="Chạy một câu hỏi")
    parser.add_argument("--test-file", default=str(Path(__file__).resolve().parents[2] / "data" / "test.json"))
    parser.add_argument("--trace-file", default=None, help="Lưu trace ra file JSON")
    parser.add_argument("--batch", action="store_true", help="Chạy batch test")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild Chroma index từ đầu")
    return parser


def _resolve_path(path_str: str, root_dir: Path) -> Path:
    """Resolve path. Nếu là relative, dùng root_dir (ko phải CWD)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (root_dir / p).resolve()


def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        output_dir = assistant.settings.traces_dir / "batch"
        test_file = _resolve_path(args.test_file, assistant.settings.root_dir)
        summary = assistant.run_batch(
            test_file=test_file,
            output_dir=output_dir,
            rebuild_index=args.rebuild_index,
        )
        print(f"Batch complete: {summary['route_ok']}/{summary['total']} route OK, "
              f"{summary['status_ok']}/{summary['total']} status OK")
        print(f"Summary saved to {output_dir / 'summary.json'}")

    elif args.question:
        trace_path = Path(args.trace_file) if args.trace_file else None
        result = assistant.ask(args.question, trace_file=trace_path, rebuild_index=args.rebuild_index)
        print(result.get("final_answer", ""))

    else:
        print("Vui lòng dùng --question hoặc --batch. Xem --help để biết thêm.")


if __name__ == "__main__":
    main()
