"""Offline checks for Slack's speaker-aware participation gate (no live APIs)."""
import asyncio
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaic import compat
from isaic import core
from isaic.platform import Message as PlatformMessage
try:
    from isaic.slack_adapter import SlackAdapter
except ModuleNotFoundError as exc:
    if exc.name not in {"slack_bolt", "slack_bolt.adapter", "slack_bolt.async_app"}:
        raise
    # The upstream Hydra environment has all model dependencies but not the
    # downstream-only Slack SDK. Stub only the two import-time classes; the
    # adapter method under test (_build_message) does not instantiate either.
    for name in (
        "slack_bolt", "slack_bolt.adapter", "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.aiohttp", "slack_bolt.async_app",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["slack_bolt.adapter.socket_mode.aiohttp"].AsyncSocketModeHandler = type(
        "AsyncSocketModeHandler", (), {}
    )
    sys.modules["slack_bolt.async_app"].AsyncApp = type("AsyncApp", (), {})
    from isaic.slack_adapter import SlackAdapter

B = core.IsaicBot
ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")


NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


print("ParticipationState transitions")
state = core.ParticipationState()
state.observe("user:U1", "Alice", NOW, "100.001", 6)
check("one speaker is session", state.auto_mode(NOW, 6), "session")
state.observe("user:U2", "Bob", NOW + timedelta(minutes=1), "100.002", 6)
check("second speaker triggers ambient", state.auto_mode(NOW + timedelta(minutes=1), 6), "ambient")
check(
    "resume is six hours after multi-speaker evidence",
    state.session_resumes_at(NOW + timedelta(minutes=1), 6),
    NOW + timedelta(hours=6, minutes=1),
)
state.observe("user:U2", "Bob", NOW + timedelta(hours=5), "100.003", 6)
check(
    "lone speaker does not refresh hold",
    state.session_resumes_at(NOW + timedelta(hours=5), 6),
    NOW + timedelta(hours=6, minutes=1),
)
check("ambient expires", state.auto_mode(NOW + timedelta(hours=6, minutes=1), 6), "session")

distant = core.ParticipationState()
distant.observe("user:U1", "Alice", NOW, "100.001", 6)
distant.observe("user:U2", "Bob", NOW + timedelta(minutes=16), "100.002", 6)
check("outside activation window stays session", distant.auto_mode(NOW + timedelta(minutes=16), 6), "session")

switching = core.ParticipationState()
switching.observe("user:U1", "Alice", NOW, "100.001", 6)
switching.observe("user:U2", "Bob", NOW + timedelta(minutes=1), "100.002", 6)
switching.observe("user:U1", "Alice", NOW + timedelta(hours=5), "100.003", 6)
check("speaker switch renews hold", switching.session_resumes_at(NOW + timedelta(hours=5), 6), NOW + timedelta(hours=11))

reply_state = core.ParticipationState()
reply_state.observe(
    "user:U2", "Bob", NOW, "100.010", 6,
    reply_target=("user:U1", "Alice"),
)
check("cross-human reply triggers ambient", reply_state.auto_mode(NOW, 6), "ambient")
roundtrip = core.ParticipationState.from_dict(reply_state.to_dict())
check("state persistence keys", set(roundtrip.speaker_last_seen), {"user:U1", "user:U2"})
check("state persistence mode", roundtrip.auto_mode(NOW, 6), "ambient")
check("malformed persistence fails soft", core.ParticipationState.from_dict("oops").speaker_last_seen, {})

fd, state_path = tempfile.mkstemp(suffix=".json")
os.close(fd)
try:
    usage = core.ModelProvider(
        name="Haiku participation gate", model_id="claude-haiku-4-5",
        input_cost_per_million=1.0, output_cost_per_million=5.0,
    )
    usage.record_usage(10, 2)
    usage.total_requests = 1
    manager = core.ConversationManager()
    manager.participation_states["C123"] = reply_state
    manager.save_memories(state_path, providers=[usage])
    loaded_usage = core.ModelProvider(
        name="Haiku participation gate", model_id="claude-haiku-4-5",
        input_cost_per_million=1.0, output_cost_per_million=5.0,
    )
    loaded = core.ConversationManager()
    loaded.load_memories(state_path, providers=[loaded_usage])
    check("manager persists participation", loaded.participation_states["C123"].auto_mode(NOW, 6), "ambient")
    check("manager persists gate stats", loaded_usage.total_requests, 1)
finally:
    os.remove(state_path)


def fake_message(author_id="U1", name="Alice", bot_account=False, **extra):
    author = types.SimpleNamespace(id=author_id, display_name=name, bot=bot_account)
    values = dict(
        id=extra.pop("id", "1786200000.001"),
        author=author,
        webhook_id=None,
        content=extra.pop("content", "hello"),
        attachments=extra.pop("attachments", []),
        mentions=extra.pop("mentions", []),
        created_at=extra.pop("created_at", NOW),
        reference=extra.pop("reference", None),
        is_self=extra.pop("is_self", False),
    )
    values.update(extra)
    return types.SimpleNamespace(**values)


print("Slack identity, mentions, timestamps, and backchannels")
normal = fake_message(author_id="U42", name="Alice")
real_bot = fake_message(author_id="B8", name="OtherBot", bot_account=True)
check("normal identity uses Slack user id", B._speaker_identity(normal), ("user:U42", "Alice"))
check("real bot is not a speaker", B._speaker_identity(real_bot), None)
for text in ("Certainly.", "lol", "sounds good", ":thumbsup:"):
    check(f"backchannel {text!r}", B._is_obvious_backchannel(text), True)
check("question is not a backchannel", B._is_obvious_backchannel("Could you check this?"), False)
check("attachment is not auto-silenced", B._is_obvious_backchannel("nice", True), False)


class TinyPlatform:
    bot_user_id = "UBOT"
    team_id = "T1"


pm = PlatformMessage(
    id="1786200000.123456", channel_id="C1", thread_id=None,
    author_id="U1", author_name="Alice", text="hello", mentions_bot=True,
)
wrapped = compat.Message(TinyPlatform(), pm)
check("compat preserves mention", [m.id for m in wrapped.mentions], ["UBOT"])
check("Slack timestamp becomes aware UTC", wrapped.created_at.tzinfo, timezone.utc)


async def adapter_checks():
    adapter = object.__new__(SlackAdapter)
    adapter.bot_user_id = "UBOT"

    async def resolve(uid):
        return {"U1": "Alice"}.get(uid, uid)

    adapter.resolve_user_name = resolve
    built = await SlackAdapter._build_message(adapter, {
        "ts": "1786200000.200000", "channel": "C1", "user": "U1",
        "text": "<@UBOT> can you check this?",
    })
    check("adapter records mention before stripping", built.mentions_bot, True)
    check("adapter still strips leading mention", built.text, "can you check this?")

    class HistoryPlatform(TinyPlatform):
        def __init__(self, history):
            self.history = history

        async def fetch_history(self, *_args, **_kwargs):
            return self.history

    current_pm = PlatformMessage(
        id="1786200002.000000", channel_id="C1", thread_id="1786200000.000000",
        author_id="U2", author_name="Bob", text="following up",
    )
    bot_pm = PlatformMessage(
        id="1786200001.000000", channel_id="C1", thread_id="1786200000.000000",
        author_id="UBOT", author_name="ISAIC", text="bot answer",
        is_bot=True, is_self=True,
    )
    human_pm = PlatformMessage(
        id="1786200000.000000", channel_id="C1", thread_id="1786200000.000000",
        author_id="U1", author_name="Alice", text="root question",
    )
    platform = HistoryPlatform([human_pm, bot_pm, current_pm])
    obj = object.__new__(B)
    obj.platform = platform
    obj.participation_history_limit = 500
    current = compat.Message(platform, current_pm)
    target = await B._resolve_reply_target(obj, current)
    check("thread target is immediately preceding bot turn", (target.author.id, target.is_self), ("UBOT", True))

    platform.history = [human_pm, current_pm]
    target = await B._resolve_reply_target(obj, current)
    check("thread target falls back to preceding human turn", target.author.id, "U1")


def gate_stub(classifier_result=(False, 0.1, "humans_conversing")):
    obj = object.__new__(B)
    obj.user = types.SimpleNamespace(id="UBOT")
    obj.participation_enabled = True
    obj.participation_channel_modes = {}
    obj.participation_solo_reset_hours = 6.0
    obj.participation_cooldown_minutes = 15.0
    obj._participation_classifier_locks = {}
    obj.manager = types.SimpleNamespace(mark_dirty=lambda: None)

    async def classify(_message):
        return classifier_result

    obj._classify_ambient_intervention = classify
    return obj


async def decision_checks():
    print("Participation decisions")
    ambient = core.ParticipationState()
    ambient.observe("user:U1", "Alice", NOW, "1", 6)
    ambient.observe("user:U2", "Bob", NOW, "2", 6)
    obj = gate_stub()

    msg = fake_message(author_id="U2", id="2", name="Bob", content="Certainly.")
    got = await B._should_participate(obj, msg, "C1", ambient, None, False)
    check("ambient acknowledgement silent", (got.should_respond, got.reason), (False, "obvious_backchannel"))

    mentioned = fake_message(
        author_id="U2", id="2", name="Bob", content="hey bot",
        mentions=[types.SimpleNamespace(id="UBOT")],
    )
    got = await B._should_participate(obj, mentioned, "C1", ambient, None, False)
    check("mention bypass", (got.should_respond, got.reason), (True, "bot_mention"))

    got = await B._should_participate(obj, msg, "C1", ambient, None, True)
    check("prefix bypass", got.should_respond, True)

    bot_target = fake_message(author_id="UBOT", name="ISAIC", bot_account=True, is_self=True)
    got = await B._should_participate(obj, msg, "C1", ambient, bot_target, False)
    check("reply after bot turn bypass", got.should_respond, True)

    human_target = fake_message(author_id="U1", name="Alice")
    discussion = fake_message(author_id="U2", id="2", name="Bob", content="I agree with that analysis")
    got = await B._should_participate(obj, discussion, "C1", ambient, human_target, False)
    check("human thread reply hard silent", (got.should_respond, got.reason), (False, "reply_to_human"))

    single = core.ParticipationState()
    single.observe("user:U2", "Bob", NOW, "2", 6)
    got = await B._should_participate(obj, discussion, "C1", single, None, False)
    check("single speaker session replies", got.should_respond, True)

    obj.participation_channel_modes["C1"] = "tags"
    got = await B._should_participate(obj, discussion, "C1", single, None, False)
    check("manual tags silent", (got.should_respond, got.reason), (False, "tag_only_mode"))
    obj.participation_channel_modes.clear()

    approving = gate_stub((True, 0.96, "important_correction"))
    got = await B._should_participate(approving, discussion, "C1", ambient, None, False)
    check("high-value Haiku approval", (got.should_respond, got.unsolicited), (True, True))
    check("approval starts cooldown", ambient.last_unsolicited_reply_at, NOW)

    later = fake_message(
        author_id="U2", id="3", name="Bob", content="A different observation",
        created_at=NOW + timedelta(minutes=2),
    )
    ambient.latest_human_message_id = "3"
    ambient.latest_human_message_at = later.created_at
    got = await B._should_participate(approving, later, "C1", ambient, None, False)
    check("unsolicited cooldown", (got.should_respond, got.reason), (False, "unsolicited_reply_cooldown"))

    stale = core.ParticipationState()
    stale.observe("user:U1", "Alice", NOW, "1", 6)
    stale.observe("user:U2", "Bob", NOW, "99", 6)
    old = fake_message(author_id="U1", id="1", name="Alice", content="Substantive but stale")
    got = await B._should_participate(obj, old, "C1", stale, None, False)
    check("stale turn suppressed", (got.should_respond, got.reason), (False, "conversation_advanced"))


async def classifier_checks():
    print("Haiku JSON + usage accounting")

    class FakeMessages:
        def __init__(self, payload):
            self.payload = payload

        def create(self, **_kwargs):
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=120, output_tokens=18),
                content=[types.SimpleNamespace(type="text", text=self.payload)],
            )

    obj = object.__new__(B)
    obj.participation_classifier_model = "claude-haiku-4-5"
    obj.participation_classifier_threshold = 0.9
    obj.participation_client = types.SimpleNamespace(
        messages=FakeMessages('{"action":"respond","intervention_value":0.93,"reason":"urgent correction"}')
    )
    obj.participation_usage = core.ModelProvider(
        name="Haiku participation gate", model_id="claude-haiku-4-5",
        input_cost_per_million=1.0, output_cost_per_million=5.0,
    )
    dirty = []
    obj.manager = types.SimpleNamespace(mark_dirty=lambda: dirty.append(True))

    async def context(_message):
        return "Alice: claim\nBob: correction?"

    obj._ambient_classifier_context = context
    got = await B._classify_ambient_intervention(obj, fake_message())
    check("valid approval parsed", got, (True, 0.93, "urgent_correction"))
    check("classifier request counted", obj.participation_usage.total_requests, 1)
    check("classifier tokens counted", (obj.participation_usage.total_input_tokens, obj.participation_usage.total_output_tokens), (120, 18))
    check("classifier usage dirties persistence", bool(dirty), True)

    obj.participation_client.messages.payload = '{"action":"respond","intervention_value":0.70,"reason":"merely interesting"}'
    got = await B._classify_ambient_intervention(obj, fake_message())
    check("below threshold stays silent", got, (False, 0.7, "merely_interesting"))


async def main():
    await adapter_checks()
    await decision_checks()
    await classifier_checks()


asyncio.run(main())
print(f"\n{ok} passed, {fail} failed")
if fail:
    raise SystemExit(1)
