"""Offline checks: images must never reach a non-user turn (2026-11-08).

The live failure this pins. At 08:26 every turn in a channel started 400ing on
both heads at once:

    [Deepseek] 400 - Image in assistant message is not supported
    [Claude]   400 - messages.9.content: 'image' blocks are not permitted
                     within assistant turns

Root cause: fetch_thread_history attaches any image attachment it finds and
roles the message by AUTHOR, so one of our OWN bot messages carrying an image
(a LaTeX render, say) becomes an illegal assistant image block -- and then
every subsequent turn in that channel fails until it scrolls out of the
history window. It was latent for the shim providers only because
_strip_images_from_messages removed image blocks for supports_vision=False
providers; flipping Deepseek's vision on removed that accidental shield.
Claude never had one.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_assistant_images.py
"""
import asyncio
import copy
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))
BotClass = getattr(bot, "ClaudeBot", None) or getattr(bot, "IsaicBot")

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def img(data="AAAA"):
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": data}}


def has_image(msgs, role=None):
    for m in msgs:
        if role is not None and m.get("role") != role:
            continue
        c = m.get("content")
        if isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "image" for b in c):
            return True
    return False


strip = bot._strip_assistant_images

print("the invariant")
out = strip([{"role": "assistant", "content": [img()]}])
check("an image alone in an assistant turn becomes a text placeholder",
      not has_image(out) and isinstance(out[0]["content"], str)
      and "Image omitted" in out[0]["content"])

out = strip([{"role": "assistant",
              "content": [{"type": "text", "text": "here you go"}, img()]}])
check("a mixed assistant turn keeps its text and loses only the image",
      not has_image(out)
      and out[0]["content"][0]["text"] == "here you go"
      and out[0]["content"][1]["type"] == "text")

out = strip([{"role": "assistant", "content": [img("A"), img("B"), img("C")]}])
check("several images are all replaced",
      not has_image(out) and len(out[0]["content"]) == 3)

print("\nuser turns are untouched — this is where images are legal")
user = [{"role": "user", "content": [{"type": "text", "text": "look"}, img()]}]
out = strip(user)
check("a user image survives", has_image(out, "user"))
check("and the turn is returned unchanged, not rebuilt", out[0] is user[0])

print("\nshapes it must not disturb")
plain = [{"role": "assistant", "content": "just text"}]
check("a string-content turn passes through", strip(plain)[0] is plain[0])
textonly = [{"role": "assistant",
             "content": [{"type": "text", "text": "a"},
                         {"type": "text", "text": "b"}]}]
check("an image-free list turn passes through untouched",
      strip(textonly)[0] is textonly[0])
check("an empty list is harmless", strip([{"role": "assistant",
                                           "content": []}])[0]["content"] == [])
check("a non-list, non-string content is left alone",
      strip([{"role": "assistant", "content": None}])[0]["content"] is None)

print("\nmetadata and immutability")
src = [{"role": "assistant", "content": [img()], "_msg_id": 4242}]
out = strip(src)
check("_msg_id survives the rewrite (the reasoning cache keys on it)",
      out[0].get("_msg_id") == 4242)
check("the caller's message is NOT mutated in place",
      src[0]["content"][0]["type"] == "image")

print("\nordering and count are preserved")
convo = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [img()]},
    {"role": "user", "content": [{"type": "text", "text": "and this"}, img()]},
    {"role": "assistant", "content": "ok"},
]
out = strip(convo)
check("every turn comes back, in order",
      len(out) == 4 and [m["role"] for m in out]
      == ["user", "assistant", "user", "assistant"])
check("only the assistant image went",
      not has_image(out, "assistant") and has_image(out, "user"))

print("\nend to end through the real fetch_thread_history")


class _Author:
    def __init__(self, aid, is_bot, name):
        self.id = aid
        self.bot = is_bot
        self.display_name = name


class _Attachment:
    def __init__(self, filename, size=1024):
        self.filename = filename
        self.size = size
        self.url = "https://example.invalid/" + filename


class _Msg:
    def __init__(self, mid, author, content="", attachments=()):
        self.id = mid
        self.author = author
        self.content = content
        self.attachments = list(attachments)
        self.webhook_id = None
        self.reference = None
        self.embeds = []


class _State:
    def __init__(self, me):
        self.user = me


class _Channel:
    def __init__(self, msgs, me):
        self._msgs = msgs
        self._state = _State(me)

    def history(self, limit=100, oldest_first=False):
        # Discord yields newest first; fetch_thread_history reverses.
        msgs = list(reversed(self._msgs))[:limit]

        async def _gen():
            for m in msgs:
                yield m
        return _gen()


ME = _Author(1, True, "Ouroboropus")
HUMAN = _Author(2, False, "viksalos")

# The real shape of the outage: our own bot posted a rendered PNG, and from
# then on every turn carried an illegal assistant image.
channel = _Channel([
    _Msg(1, HUMAN, "render x^2 for me"),
    _Msg(2, ME, "[Claude] here it is", [_Attachment("latex_render.png")]),
    _Msg(3, HUMAN, "salam aleykum"),
], ME)

m = bot.ConversationManager.__new__(bot.ConversationManager)
m.reading_materials = {}
hist = asyncio.run(bot.ConversationManager.fetch_thread_history(m, channel))
check("history came back", len(hist) == 3)
check("NO image block in any assistant turn", not has_image(hist, "assistant"))
check("no image block anywhere, since only the bot posted one",
      not has_image(hist))
blob = str(hist)
check("the fact that an image was posted is still recorded",
      "latex_render.png" in blob)
check("and the bot's own text survives beside it", "here it is" in blob)

print("\nand the shim can no longer emit an assistant image_url")
b = BotClass.__new__(BotClass)
ds = copy.deepcopy(bot.DEEPSEEK_PROVIDER)
check("Deepseek really is vision-enabled now (the flip that exposed this)",
      ds.supports_vision is True)
# Vision-capable providers skip _strip_images_from_messages entirely, so the
# sanitised history is all that stands between us and the 400.
conv = BotClass._convert_messages_to_openai_format(b, hist)
bad = [c for c in conv if c.get("role") == "assistant"
       and isinstance(c.get("content"), list)
       and any(p.get("type") == "image_url" for p in c["content"])]
check("no assistant turn carries an image_url after conversion", not bad)
# Prove the test would catch a regression: unsanitised history still would.
poisoned = BotClass._convert_messages_to_openai_format(
    b, [{"role": "assistant", "content": [img()]}])
check("the check is live — unsanitised input DOES produce one",
      poisoned[0]["content"][0]["type"] == "image_url")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
