"""SlackAdapter — a ChatPlatform implementation on slack_bolt (Socket Mode).

No public URL is needed: events arrive over a WebSocket (App-Level token). The
adapter normalizes Slack events into platform.Message / platform.Reaction and
exposes send_text/send_file/fetch_history/typing/make_thread back to the core.

Commands use the `!`-prefix text-parse path (not slash commands) so the whole
stacked grammar (`!think:max !judah …`) keeps working through one code path.
The bot must be invited to a channel (`/invite @ISAIC`) to receive its messages.
"""
from __future__ import annotations

import asyncio
import io
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from .platform import Attachment, Message, Reaction

# Slack delivers reactions as shortnames; map the ~handful the calibration code
# recognizes to unicode so the core's good/bad emoji sets work unchanged.
SLACK_EMOJI_TO_UNICODE = {
    "+1": "👍", "thumbsup": "👍",
    "heart": "❤️", "fire": "🔥",
    "white_check_mark": "✅", "heavy_check_mark": "✅",
    "joy": "😂", "sparkling_heart": "💖", "100": "💯",
    "-1": "👎", "thumbsdown": "👎",
    "x": "❌", "negative_squared_cross_mark": "❌", "confused": "😕",
}
# Reverse map for add_reaction (core emits unicode from [react: …]); best effort.
_UNICODE_TO_SLACK = {
    "👍": "+1", "❤️": "heart", "🔥": "fire", "✅": "white_check_mark",
    "😂": "joy", "💖": "sparkling_heart", "💯": "100", "👎": "-1",
    "❌": "x", "😕": "confused", "🎉": "tada", "👀": "eyes", "🤔": "thinking_face",
    "🙏": "pray", "🚀": "rocket", "✨": "sparkles", "💡": "bulb",
}

_SEEN_CAP = 2000
_TYPING_PLACEHOLDER = "💭 _thinking…_"  # posted/deleted by _typing_cm; filtered from history


class SlackAdapter:
    """Implements isaic.platform.ChatPlatform on Bolt's AsyncApp + Socket Mode."""

    def __init__(self, bot_token: str, app_token: str):
        self._bot_token = bot_token
        self._app_token = app_token
        # Socket Mode needs no signing secret; skip Bolt's construction-time
        # token verification (we do an explicit auth_test() in start()).
        self.app = AsyncApp(token=bot_token, token_verification_enabled=False)
        self.bot_user_id = ""
        self.team_id = ""
        self._msg_handler = None
        self._rxn_handler = None
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._user_names: dict[str, str] = {}
        self._register_handlers()

    # ---- handler registration -------------------------------------------
    def on_message(self, handler) -> None:
        self._msg_handler = handler

    def on_reaction(self, handler) -> None:
        self._rxn_handler = handler

    def auth_headers(self) -> dict:
        """Header for downloading private Slack file URLs."""
        return {"Authorization": f"Bearer {self._bot_token}"}

    def _dedupe(self, key) -> bool:
        """True if this event was already seen (Socket Mode redelivery)."""
        if key is None:
            return False
        key = str(key)
        if key in self._seen:
            return True
        self._seen[key] = None
        while len(self._seen) > _SEEN_CAP:
            self._seen.popitem(last=False)
        return False

    def _register_handlers(self) -> None:
        app = self.app

        @app.event("message")
        async def _on_message(event, body, logger=None):  # noqa: ANN001
            await self._dispatch_message(event, body)

        @app.event("app_mention")
        async def _on_mention(event, body, logger=None):  # noqa: ANN001
            # The `message` event already covers @-mentions in channels the bot
            # is in; handle here only to ack and avoid an "unhandled" warning.
            return

        @app.event("reaction_added")
        async def _on_reaction_added(event, body, logger=None):  # noqa: ANN001
            await self._dispatch_reaction(event, body, removed=False)

        @app.event("reaction_removed")
        async def _on_reaction_removed(event, body, logger=None):  # noqa: ANN001
            await self._dispatch_reaction(event, body, removed=True)

        @app.command("/isaic")
        async def _isaic_cmd(ack, respond):  # noqa: ANN001
            await ack()
            await respond("ISAIC is here. In a channel I'm a member of, type `!help` "
                          "for commands, or just @-mention me. Invite me with `/invite @ISAIC`.")

    # ---- inbound dispatch ------------------------------------------------
    async def _dispatch_message(self, event: dict, body: dict) -> None:
        subtype = event.get("subtype")
        # Allow plain messages + file shares; drop edits/deletes/joins/etc.
        if subtype not in (None, "file_share"):
            return
        if event.get("bot_id"):
            return  # ignore other bots AND our own posts (we read ours via history)
        user = event.get("user")
        if not user or user == self.bot_user_id:
            return
        key = body.get("event_id") or event.get("client_msg_id") or f"{event.get('channel')}:{event.get('ts')}"
        if self._dedupe(key):
            return
        msg = await self._build_message(event)
        if self._msg_handler:
            asyncio.create_task(self._safe(self._msg_handler(msg)))

    async def _dispatch_reaction(self, event: dict, body: dict, *, removed: bool) -> None:
        if self._rxn_handler is None:
            return
        item = event.get("item", {}) or {}
        # Fall back to a per-(message,emoji,user) key if event_id is absent, so a
        # missing id never collapses to a single global '-a'/'-r' dedupe slot.
        base = body.get("event_id") or f"{item.get('ts')}:{event.get('reaction')}:{event.get('user')}"
        if self._dedupe(base + ("-r" if removed else "-a")):
            return
        shortname = (event.get("reaction") or "").split("::")[0]  # strip skin-tone variants
        emoji = SLACK_EMOJI_TO_UNICODE.get(shortname, shortname)
        rxn = Reaction(
            emoji=emoji,
            message_id=item.get("ts", ""),
            channel_id=item.get("channel", ""),
            user_id=event.get("user", ""),
            item_user_id=event.get("item_user"),
            removed=removed,
        )
        asyncio.create_task(self._safe(self._rxn_handler(rxn)))

    async def _safe(self, coro) -> None:
        try:
            await coro
        except Exception as e:  # don't let a handler crash kill the socket
            print(f"⚠️  handler error: {type(e).__name__}: {e}")

    async def _build_message(self, event: dict) -> Message:
        channel = event.get("channel", "")
        text = self._strip_leading_mention(event.get("text", "") or "")
        files = event.get("files", []) or []
        return Message(
            id=event.get("ts", ""),
            channel_id=channel,
            thread_id=event.get("thread_ts"),
            author_id=event.get("user", ""),
            author_name=await self.resolve_user_name(event.get("user", "")),
            text=text,
            attachments=[self._to_attachment(f) for f in files],
            is_bot=bool(event.get("bot_id")),
            is_self=event.get("user") == self.bot_user_id,
            platform="slack",
            reply_to_text=None,
            raw=event,
        )

    def _strip_leading_mention(self, text: str) -> str:
        m = f"<@{self.bot_user_id}>"
        if self.bot_user_id and text.lstrip().startswith(m):
            return text.lstrip()[len(m):].lstrip()
        return text

    def _to_attachment(self, f: dict) -> Attachment:
        mt = f.get("mimetype", "") or ""
        return Attachment(
            filename=f.get("name", "file"),
            url=f.get("url_private_download") or f.get("url_private", ""),
            mimetype=mt,
            size=f.get("size", 0) or 0,
            is_image=mt.startswith("image/"),
        )

    # ---- outbound + queries ---------------------------------------------
    async def send_text(self, channel_id: str, text: str, *, thread_id: Optional[str] = None) -> str:
        resp = await self.app.client.chat_postMessage(
            channel=channel_id, text=text or " ", thread_ts=thread_id, mrkdwn=True)
        return resp.get("ts", "")

    async def send_file(self, channel_id: str, data: bytes, filename: str, *,
                        thread_id: Optional[str] = None, title: Optional[str] = None) -> None:
        await self.app.client.files_upload_v2(
            channel=channel_id, thread_ts=thread_id,
            file=io.BytesIO(data), filename=filename, title=title or filename)

    async def fetch_history(self, channel_id: str, *, thread_id: Optional[str] = None,
                            limit: int = 60) -> list[Message]:
        client = self.app.client
        if thread_id:
            resp = await client.conversations_replies(channel=channel_id, ts=thread_id, limit=limit)
        else:
            resp = await client.conversations_history(channel=channel_id, limit=limit)
        raw = resp.get("messages", []) or []
        out: list[Message] = []
        for m in raw:
            st = m.get("subtype")
            if st not in (None, "file_share", "bot_message", "thread_broadcast"):
                continue
            # Skip a stray "thinking…" placeholder if a chat_delete ever failed,
            # so it never leaks back into the model's context as an unlabeled turn.
            if (m.get("text", "") or "").strip() == _TYPING_PLACEHOLDER:
                continue
            is_bot = bool(m.get("bot_id"))
            uid = m.get("user", "") or ""
            is_self = uid == self.bot_user_id or is_bot
            name = "assistant" if is_self else await self.resolve_user_name(uid)
            out.append(Message(
                id=m.get("ts", ""),
                channel_id=channel_id,
                thread_id=thread_id,
                author_id=uid or (m.get("bot_id", "") if is_bot else ""),
                author_name=name,
                text=m.get("text", "") or "",
                attachments=[self._to_attachment(f) for f in (m.get("files", []) or [])],
                is_bot=is_bot,
                is_self=is_self,
                platform="slack",
                raw=m,
            ))
        if not thread_id:
            out.reverse()  # conversations_history is newest-first
        return out

    async def fetch_thread_index(self, channel_id: str, *, limit: int = 5) -> str:
        return ""  # handled (stubbed) in core's ConversationManager

    def typing(self, channel_id: str, *, thread_id: Optional[str] = None):
        return self._typing_cm(channel_id, thread_id)

    @asynccontextmanager
    async def _typing_cm(self, channel_id: str, thread_id: Optional[str]):
        ts = None
        try:
            resp = await self.app.client.chat_postMessage(
                channel=channel_id, text=_TYPING_PLACEHOLDER, thread_ts=thread_id, mrkdwn=True)
            ts = resp.get("ts")
        except Exception:
            ts = None
        try:
            yield
        finally:
            if ts:
                try:
                    await self.app.client.chat_delete(channel=channel_id, ts=ts)
                except Exception:
                    pass

    async def make_thread(self, channel_id: str, name: str, anchor_id: Optional[str] = None) -> str:
        # Slack threads aren't created explicitly; replies hang off the anchor ts.
        return anchor_id or ""

    async def resolve_user_name(self, user_id: str) -> str:
        if not user_id:
            return "someone"
        if user_id in self._user_names:
            return self._user_names[user_id]
        try:
            resp = await self.app.client.users_info(user=user_id)
            prof = resp.get("user", {}).get("profile", {}) or {}
            name = (prof.get("display_name") or prof.get("real_name")
                    or resp.get("user", {}).get("name") or user_id)
        except Exception:
            name = user_id
        self._user_names[user_id] = name
        return name

    async def download_file(self, attachment: Attachment) -> Optional[bytes]:
        if not attachment.url:
            return None
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url, headers=self.auth_headers()) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()

    async def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        name = emoji.strip().strip(":")
        if emoji in _UNICODE_TO_SLACK:
            name = _UNICODE_TO_SLACK[emoji]
        try:
            await self.app.client.reactions_add(channel=channel_id, timestamp=message_id, name=name)
        except Exception:
            pass  # unknown/duplicate reaction names are non-fatal

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        auth = await self.app.client.auth_test()
        self.bot_user_id = auth.get("user_id", "")
        self.team_id = auth.get("team_id", "")
        print(f"✅ ISAIC connected as {auth.get('user')} ({self.bot_user_id}) · team {self.team_id}")
        handler = AsyncSocketModeHandler(self.app, self._app_token)
        await handler.start_async()
