"""Compatibility layer — a thin discord.py-shaped surface backed by a ChatPlatform.

`core.py` is a faithful port of the original Discord bot. Rather than rewrite all
~86 message/channel call sites, core imports THIS module as `discord`:

    from . import compat as discord

so `discord.File`, `discord.Embed`, `message.content`, `message.channel.send(...)`,
`channel.typing()`, `channel.history(...)`, `isinstance(x, discord.Thread)` → the
shims below, all of which route real I/O through the injected `ChatPlatform`
(Slack). No discord.py dependency exists.

IDs are strings (Slack ts / C…/U…/T… ids). A "channel" here may represent a
channel or a thread; threads share their parent channel's id and carry a
`thread_id` (Slack `thread_ts`).
"""
from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, Optional

from . import formatting
from .platform import Attachment as _PAttachment
from .platform import ChatPlatform
from .platform import Message as _PMessage


# ---- exceptions -------------------------------------------------------------
class PlatformError(Exception):
    """Stand-in for discord.HTTPException."""


class NotFound(PlatformError):
    """Stand-in for discord.NotFound."""


HTTPException = PlatformError


# ---- enums / misc discord.X used only as values or in annotations ----------
class MessageType:
    default = "default"
    reply = "reply"


class _Color:
    @staticmethod
    def blue() -> str:
        return "#3b82f6"

    @staticmethod
    def green() -> str:
        return "#22c55e"

    @staticmethod
    def red() -> str:
        return "#ef4444"

    @staticmethod
    def gold() -> str:
        return "#eab308"


Color = _Color


class Embed:
    """Minimal Embed → rendered to Slack mrkdwn on send."""

    def __init__(self, title: str = None, description: str = None,
                 color: Any = None, url: str = None, **_):
        self.title = title
        self.description = description
        self.color = color
        self.url = url
        self.fields: list[tuple[str, str, bool]] = []
        self.footer: Optional[str] = None

    def add_field(self, *, name: str, value: str, inline: bool = False) -> "Embed":
        self.fields.append((name, value, inline))
        return self

    def set_footer(self, *, text: str = None, **_) -> "Embed":
        self.footer = text
        return self

    def to_markdown(self) -> str:
        # Emit Discord-flavored Markdown (**bold**); OutChannel.send runs it
        # through md_to_mrkdwn exactly once, so bold stays bold on Slack (do NOT
        # pre-convert to single-asterisk here, or it becomes italic).
        parts: list[str] = []
        if self.title:
            parts.append(f"**{self.title}**")
        if self.description:
            parts.append(self.description)
        for name, value, _inline in self.fields:
            parts.append(f"**{name}**\n{value}")
        if self.footer:
            parts.append(f"_{self.footer}_")
        return "\n\n".join(parts)

    # Back-compat alias.
    to_mrkdwn = to_markdown


class File:
    """Stand-in for discord.File. Holds bytes + a filename; uploaded via the
    platform's send_file on the receiving channel."""

    def __init__(self, fp: Any, filename: str = None, **_):
        if isinstance(fp, (bytes, bytearray)):
            self.data = bytes(fp)
        elif hasattr(fp, "read"):
            pos = fp.tell() if hasattr(fp, "tell") else None
            self.data = fp.read()
            if pos is not None:
                try:
                    fp.seek(pos)
                except Exception:
                    pass
        else:
            self.data = bytes(fp)
        self.filename = filename or "file"


class Thread:  # marker only; isinstance(x, discord.Thread) is rewritten to x.is_thread
    pass


class Reaction:  # annotation-only stand-in
    pass


class User:  # annotation-only stand-in
    pass


class Intents:  # unused at runtime; kept so any stray reference resolves
    @staticmethod
    def default() -> "Intents":
        return Intents()


abc = SimpleNamespace(Messageable=object)


# ---- the working shims: author / attachment / message / channel ------------
class _Author:
    def __init__(self, uid: str, display_name: str, bot: bool):
        self.id = uid
        self.display_name = display_name
        self.name = display_name
        self.bot = bot


class _Attachment:
    def __init__(self, platform: ChatPlatform, att: _PAttachment):
        self._platform = platform
        self._att = att
        self.filename = att.filename
        self.url = att.url
        self.size = att.size or 0
        self.content_type = att.mimetype

    async def read(self) -> Optional[bytes]:
        return await self._platform.download_file(self._att)


class OutMessage:
    """The lightweight object returned by channel.send() — only .id / .channel
    are read by the core (reasoning-cache keying)."""

    def __init__(self, channel: "OutChannel", ts: Optional[str]):
        self.id = ts
        self.channel = channel


class OutChannel:
    """A channel or thread to read history from / send to, backed by the platform."""

    def __init__(self, platform: ChatPlatform, channel_id: str,
                 thread_id: Optional[str] = None, is_thread: bool = False):
        self._platform = platform
        self.id = channel_id
        self.thread_id = thread_id
        self.is_thread = is_thread
        # threads share the parent channel's id on Slack
        self.parent_id = channel_id if is_thread else None
        # source reads channel._state.user.id to identify the bot's own messages
        self._state = SimpleNamespace(
            user=SimpleNamespace(id=getattr(platform, "bot_user_id", "")))

    async def send(self, content: str = None, *, embed: Embed = None,
                   file: File = None, files: list = None, **_) -> OutMessage:
        text = content
        if text is None and embed is not None:
            text = embed.to_markdown()
        elif embed is not None and text is not None:
            text = text + "\n\n" + embed.to_markdown()
        first_ts: Optional[str] = None
        if text:
            # Convert Markdown→mrkdwn ONCE, then chunk the converted text.
            for chunk in formatting.chunk_for_slack(formatting.md_to_mrkdwn(text)):
                if chunk.strip():
                    ts = await self._platform.send_text(self.id, chunk, thread_id=self.thread_id)
                    if first_ts is None:
                        first_ts = ts
        all_files = list(files or [])
        if file is not None:
            all_files.append(file)
        for f in all_files[:10]:  # sanity cap
            await self._platform.send_file(self.id, f.data, f.filename, thread_id=self.thread_id)
        return OutMessage(self, first_ts)

    def typing(self):
        return self._platform.typing(self.id, thread_id=self.thread_id)

    async def history(self, limit: int = 50, **_):
        msgs = await self._platform.fetch_history(self.id, thread_id=self.thread_id, limit=limit)
        for pm in msgs:
            yield Message(self._platform, pm)

    async def fetch_message(self, mid: str) -> "Message":
        raise NotFound(mid)

    async def create_thread(self, name: str = None, **_) -> "OutChannel":
        # On Slack there's no explicit thread creation; replies thread off the
        # anchor message. If we're already a thread, reuse it.
        return self


class Message:
    """Inbound message shim (discord.Message-shaped), wrapping a platform.Message."""

    def __init__(self, platform: ChatPlatform, pm: _PMessage):
        self._platform = platform
        self._pm = pm
        self.id = pm.id
        self.content = pm.text or ""
        self.author = _Author(pm.author_id, pm.author_name, pm.is_bot or pm.is_self)
        self.channel = OutChannel(platform, pm.channel_id,
                                  thread_id=pm.thread_id, is_thread=bool(pm.thread_id))
        self.guild = SimpleNamespace(id=getattr(platform, "team_id", ""))
        self.attachments = [_Attachment(platform, a) for a in pm.attachments]
        self.webhook_id = None          # no webhook proxies on Slack
        self.type = MessageType.default
        self.is_self = pm.is_self
        # reply-to context (best effort): the platform resolves the parent text.
        if pm.reply_to_text:
            self.reference = SimpleNamespace(
                message_id="ref",
                resolved=SimpleNamespace(
                    content=pm.reply_to_text,
                    author=SimpleNamespace(display_name="someone", id="", bot=False),
                    webhook_id=None,
                ),
            )
        else:
            self.reference = None

    async def add_reaction(self, emoji: str) -> None:
        await self._platform.add_reaction(self.channel.id, self.id, emoji)

    async def create_thread(self, name: str = None, **_) -> OutChannel:
        tid = await self._platform.make_thread(self.channel.id, name or "", anchor_id=self.id)
        return OutChannel(self._platform, self.channel.id, thread_id=tid, is_thread=True)

    async def reply(self, content: str = None, **kw) -> OutMessage:
        return await self.channel.send(content, **kw)
