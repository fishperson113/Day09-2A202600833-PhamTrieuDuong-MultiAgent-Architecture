from __future__ import annotations

import re


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    """Parse policy markdown thành chunks theo cấu trúc ## H2 → ### H3 → content.

    Mỗi chunk = một H3 + toàn bộ content bên dưới nó + H2 cha.
    """
    chunks: list[dict] = []
    lines = markdown_text.split("\n")

    current_h2: str | None = None
    current_h3: str | None = None
    current_content: list[str] = []

    def flush() -> None:
        """Lưu chunk hiện tại nếu có đủ H2 + H3."""
        if current_h2 and current_h3:
            rendered = (
                f"{current_h2}\n{current_h3}\n"
                + "\n".join(current_content).strip()
            )
            chunks.append({
                "section_h2": current_h2,
                "section_h3": current_h3,
                "citation": f"policy_mock_vi.md > {current_h3}",
                "rendered_text": rendered.strip(),
            })

    for line in lines:
        h2_match = re.match(r"^##\s+(.+)", line)
        h3_match = re.match(r"^###\s+(.+)", line)

        if h2_match:
            flush()
            current_h2 = h2_match.group(1).strip()
            current_h3 = None
            current_content = []
        elif h3_match:
            flush()
            current_h3 = h3_match.group(1).strip()
            current_content = []
        else:
            # Content chỉ được gom nếu đang ở trong H3
            if current_h3:
                current_content.append(line)

    flush()  # chunk cuối cùng
    return chunks
