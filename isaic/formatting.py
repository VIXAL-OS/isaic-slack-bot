"""Slack output formatting: Markdown → mrkdwn, chunking, and Block Kit helpers.

Models emit GitHub-flavored Markdown. Slack's `mrkdwn` is different:
  *bold*  (single asterisk, not **)        _italic_
  <url|text>  (not [text](url))            ```code``` fences (no language token)
  no #/##/### headers, no real tables
`md_to_mrkdwn` does a best-effort conversion; code spans are protected so their
contents are never mangled. This is a pure module (no Slack/SDK imports) so it
can be unit-tested in isolation.
"""
from __future__ import annotations

import re

# Slack renders ~3000 chars per text block comfortably; keep margin under 4000.
SLACK_CHUNK_LIMIT = 3500

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
# Link URL allows one level of balanced parens (Wikipedia/doc URLs like
# /wiki/Foo_(bar)) instead of stopping at the first ')'.
_LINK_RE = re.compile(r"!?\[([^\]]+)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)\)")
# Bold only via ** — NOT __x__, which would mangle Python dunders (__init__) in
# prose. Models emit ** for bold; Slack bold is single *.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

_BOLD_SENTINEL = "\x00B\x00"


def _protect(text: str, pattern: re.Pattern, store: list) -> str:
    def repl(m):
        store.append(m.group(0))
        return f"\x00C{len(store) - 1}\x00"
    return pattern.sub(repl, text)


def _wrap_tables(text: str) -> str:
    """Markdown pipe tables don't render in Slack — wrap each contiguous table
    block in a code fence so columns at least stay monospaced/aligned."""
    out, block = [], []

    def flush():
        if block:
            out.append("```\n" + "\n".join(block) + "\n```")
            block.clear()

    for line in text.split("\n"):
        if _TABLE_ROW_RE.match(line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def md_to_mrkdwn(text: str) -> str:
    """Convert model Markdown to Slack mrkdwn (best effort)."""
    if not text:
        return text

    # 1. Protect fenced + inline code so we don't touch their contents.
    code_store: list[str] = []
    text = _protect(text, _FENCE_RE, code_store)
    text = _protect(text, _INLINE_CODE_RE, code_store)

    # 2. Escape mrkdwn-significant chars in the prose (NOT inside links we build).
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 3. Links: [text](url) / ![alt](url) → <url|text>, protected as a unit so
    #    the bold/italic passes below don't rewrite emphasis inside link text.
    link_store: list[str] = []

    def _link(m):
        link_store.append(f"<{m.group(2)}|{m.group(1)}>")
        return f"\x00L{len(link_store) - 1}\x00"
    text = _LINK_RE.sub(_link, text)

    # 4. Bold first (** → sentinel), then italic (* → _), then sentinel → *.
    text = _BOLD_RE.sub(lambda m: f"{_BOLD_SENTINEL}{m.group(1)}{_BOLD_SENTINEL}", text)
    text = _ITALIC_STAR_RE.sub(r"_\1_", text)
    text = text.replace(_BOLD_SENTINEL, "*")

    # Restore links verbatim.
    for i, link in enumerate(link_store):
        text = text.replace(f"\x00L{i}\x00", link)

    # 5. Headers → bold line (Slack has no headers in mrkdwn text).
    text = _HEADER_RE.sub(lambda m: f"*{m.group(1)}*", text)

    # 6. Tables → fenced code.
    text = _wrap_tables(text)

    # 7. Restore code spans. Strip a leading language token off fences
    #    (```python → ```), which Slack would otherwise render literally.
    for i, code in enumerate(code_store):
        if code.startswith("```"):
            code = re.sub(r"^```[^\n`]*\n", "```\n", code, count=1)
        text = text.replace(f"\x00C{i}\x00", code)
    return text


def chunk_for_slack(text: str, limit: int = SLACK_CHUNK_LIMIT) -> list[str]:
    """Split text into <=limit-char chunks, preferring paragraph/line/space
    boundaries. Mirrors the Discord bot's _send_response chunking with a larger
    budget."""
    if not text:
        return [""]
    chunks: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


# ---- Block Kit helpers (for !help and source citations) --------------------

def section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:3000]}}


def divider() -> dict:
    return {"type": "divider"}


def header(text: str) -> dict:
    # plain_text header, max 150 chars, no mrkdwn
    return {"type": "header", "text": {"type": "plain_text", "text": text[:150], "emoji": True}}


def sources_blocks(citations: list[dict]) -> tuple[list[dict], str]:
    """Build (blocks, fallback_text) for a 🔍 Sources card from a list of
    {url, title, snippet} dicts."""
    lines = []
    for i, c in enumerate(citations, 1):
        url = c.get("url", "")
        title = c.get("title") or url
        lines.append(f"{i}. <{url}|{title}>")
    body = "\n".join(lines) if lines else "_no sources_"
    blocks = [section(f"*🔍 Sources*\n{body}")]
    fallback = "🔍 Sources:\n" + "\n".join(c.get("url", "") for c in citations)
    return blocks, fallback
