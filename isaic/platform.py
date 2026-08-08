"""Platform abstraction (Phase 4).

`ChatPlatform` is the seam that lets the core bot talk to a chat service without
importing its SDK directly. The core consumes normalized `Message` / `Reaction`
objects and calls `send_text` / `send_file` / `fetch_history` / `typing` /
`make_thread`, never `discord.*` or `slack_sdk.*`. Only the adapter
(`slack_adapter.SlackAdapter`) knows the wire format.

IDs are STRINGS everywhere (Slack timestamps like "1719536000.001200" and IDs
like "C0123ABC" / "U0123ABC" / "T0123ABC"), not ints — the whole codebase keys
channels/threads/users/messages on strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


@dataclass
class Attachment:
    """A file attached to a message. `url` is a private Slack URL that requires
    an `Authorization: Bearer <bot token>` header to download — always fetch it
    via `ChatPlatform.download_file`, never a bare GET."""
    filename: str
    url: str
    mimetype: str = ""
    size: int = 0
    is_image: bool = False


@dataclass
class Message:
    """Platform-agnostic message. Replaces discord.Message throughout the core."""
    id: str                                   # the message ts (Slack) — unique per channel
    channel_id: str
    thread_id: Optional[str]                  # Slack thread_ts; None for a top-level message
    author_id: str                            # "U…" (user) or "B…" (bot/app)
    author_name: str                          # resolved display name
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    is_bot: bool = False                      # author is any bot/app
    is_self: bool = False                     # author is THIS bot
    mentions_bot: bool = False                # raw text explicitly mentioned THIS bot
    platform: str = "slack"
    reply_to_text: Optional[str] = None       # resolved parent/quoted text (replaces msg.reference)
    raw: Any = None                           # the original event payload, if an adapter needs it


@dataclass
class Reaction:
    """A reaction add/remove event, normalized. `emoji` is unicode (the adapter
    maps Slack shortnames like '+1' → '👍')."""
    emoji: str
    message_id: str          # ts of the message that was reacted to
    channel_id: str
    user_id: str             # who reacted
    item_user_id: Optional[str] = None   # author of the reacted-to message (Slack item_user)
    removed: bool = False


MessageHandler = Callable[[Message], Awaitable[None]]
ReactionHandler = Callable[[Reaction], Awaitable[None]]


@runtime_checkable
class ChatPlatform(Protocol):
    """The interface every platform adapter implements. The core depends only on
    this, so swapping Discord ↔ Slack is a one-line config change + a new adapter."""

    #: this bot's own user/app id, set during start(); used to ignore self-messages
    bot_user_id: str
    #: the workspace/team/guild id, set during start(); part of the persistence key
    team_id: str

    async def start(self) -> None:
        """Connect and block, dispatching events to the registered handlers."""
        ...

    def on_message(self, handler: MessageHandler) -> None:
        """Register the coroutine that handles each inbound Message."""
        ...

    def on_reaction(self, handler: ReactionHandler) -> None:
        """Register the coroutine that handles each Reaction (add/remove)."""
        ...

    async def send_text(self, channel_id: str, text: str, *,
                        thread_id: Optional[str] = None) -> str:
        """Post `text` (already Slack-mrkdwn). Returns the new message's ts."""
        ...

    async def send_file(self, channel_id: str, data: bytes, filename: str, *,
                        thread_id: Optional[str] = None, title: Optional[str] = None) -> None:
        """Upload a file (LaTeX PNG / TTS MP3 / code attachment)."""
        ...

    async def fetch_history(self, channel_id: str, *, thread_id: Optional[str] = None,
                            limit: int = 60) -> list[Message]:
        """Return recent messages (oldest→newest). For a thread, the thread's
        replies; otherwise the channel's recent messages."""
        ...

    async def fetch_thread_index(self, channel_id: str, *, limit: int = 5) -> str:
        """A short human summary of other recent threads/conversations in the
        channel (for the model's situational awareness). May return ''."""
        ...

    def typing(self, channel_id: str, *, thread_id: Optional[str] = None):
        """Async context manager that signals 'working…' for the duration of a
        turn. Slack has no bot typing API, so the adapter fakes it."""
        ...

    async def make_thread(self, channel_id: str, name: str, anchor_id: Optional[str] = None) -> str:
        """Return the thread id to reply under. On Slack this is the anchor
        message's thread_ts (or its own ts), since threads aren't created
        explicitly."""
        ...

    async def resolve_user_name(self, user_id: str) -> str:
        """Display name for a user id (cached)."""
        ...

    async def download_file(self, attachment: Attachment) -> Optional[bytes]:
        """Download a (private) attachment with the right auth header."""
        ...

    def auth_headers(self) -> dict:
        """Headers needed to fetch this platform's private file URLs (e.g.
        {'Authorization': 'Bearer xoxb-…'} for Slack)."""
        ...

    async def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """React to a message (the [react: …] feature)."""
        ...
