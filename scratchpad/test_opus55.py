"""Offline migration regressions. BOT_MODULE=isaic.core tests the Slack copy."""
import asyncio
import copy
import importlib
import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, os.getcwd())
module = importlib.import_module(os.getenv("BOT_MODULE", "bot"))
Bot = getattr(module, "ClaudeBot", None) or module.IsaicBot


def response(text="A complete summary.", reason="end_turn"):
    return NS(content=[NS(type="thinking", thinking="", signature="signed"),
                       NS(type="text", text=text)], stop_reason=reason,
              usage=NS(input_tokens=100, output_tokens=50,
                       cache_read_input_tokens=20, cache_creation_input_tokens=0))



class _StreamCM:
    """Stand-in for the SDK's MessageStreamManager context manager."""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


def fresh(responses):
    b = Bot.__new__(Bot)
    b.claude_provider = copy.deepcopy(module.CLAUDE_PROVIDER)
    b.claude_provider.enabled = True
    b.usage_providers = [b.claude_provider]
    # ONE recorder for BOTH transports. The xhigh/max tier asks for 64k
    # max_tokens, which _claude_message sends through messages.stream instead
    # of messages.create (the SDK refuses that budget non-streaming). Routing
    # the stream through the same Mock keeps every assertion below
    # transport-agnostic, and .streamed records which calls took which path.
    _api = Mock(side_effect=responses)
    _streamed = []

    def _stream(**kwargs):
        _streamed.append(kwargs)
        return _StreamCM(_api(**kwargs))

    b.claude_client = NS(messages=NS(create=_api, stream=_stream,
                                     streamed=_streamed))
    b.manager = NS(memories={1: NS(longterm=NS(add=Mock(return_value=True)),
                                 get_context_string=lambda: "")},
                   fetch_thread_history=AsyncMock(return_value=[{"role": "user", "content": "Please summarize."}]),
                   save_memories=Mock())
    b._extract_notes = lambda text, guild: text
    b._mem_key = lambda channel_id: 1  # Slack scopes memory by channel.
    return b


class Migration(unittest.TestCase):
    def test_old_dollars_survive_upgrade_new_calls_and_restart(self):
        p = copy.deepcopy(module.CLAUDE_PROVIDER)
        p.load_stats({"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                      "cached_input_tokens": 1_000_000})
        self.assertAlmostEqual(p.get_cost(), 30.5)
        energy = p.get_energy_wh()
        p.record_usage(1_000_000, 1_000_000, 1_000_000)
        self.assertAlmostEqual(p.get_cost(), 54.7)
        self.assertAlmostEqual(p.get_energy_wh(), 2 * energy)
        q = copy.deepcopy(module.CLAUDE_PROVIDER)
        q.load_stats(p.to_stats_dict())
        self.assertAlmostEqual(q.get_cost(), 54.7)
        q.record_usage(1_000_000, 0)
        self.assertAlmostEqual(q.get_cost(), 58.7)

    def test_cache_creation_read_and_output_are_all_billed(self):
        b = fresh([])
        b._record_claude_usage(NS(input_tokens=1_000_000, output_tokens=1_000_000,
            cache_read_input_tokens=1_000_000, cache_creation_input_tokens=2_000_000,
            cache_creation=NS(ephemeral_1h_input_tokens=1_000_000)))
        self.assertAlmostEqual(b.claude_provider.get_cost(), 4 + 20 + .2 + 5 + 8)
        self.assertEqual(b.claude_provider.total_input_tokens, 3_000_000)

    def test_summary_skips_thinking_and_accounts_usage(self):
        b = fresh([response()])
        msg = NS(content="!summarize test", guild=NS(id=1), channel=NS(id=1, send=AsyncMock()))
        asyncio.run(b._handle_command(msg))
        b.manager.memories[1].longterm.add.assert_called_once_with("thread_test", "A complete summary.")
        request = b.claude_client.messages.create.call_args.kwargs
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["extra_body"]["output_config"]["effort"], "low")
        self.assertGreaterEqual(request["max_tokens"], 4096)
        self.assertEqual(b.claude_provider.total_requests, 1)

    def test_incomplete_or_refused_summary_is_not_saved(self):
        for text, reason in [("Partial", "max_tokens"), ("Declined", "refusal"), ("", "end_turn")]:
            with self.subTest(reason=reason):
                b = fresh([response(text, reason)])
                msg = NS(content="!summarize test", guild=NS(id=1), channel=NS(id=1, send=AsyncMock()))
                asyncio.run(b._handle_command(msg))
                b.manager.memories[1].longterm.add.assert_not_called()
                b.manager.save_memories.assert_called_once()
                self.assertGreater(b.claude_provider.get_cost(), 0)

    def test_plain_and_thinking_requests_preserve_tool_signatures(self):
        for thinking, effort in [(False, None), (True, "high"), (True, "max")]:
            with self.subTest(thinking=thinking, effort=effort):
                first = response("", "tool_use")
                first.content.append(NS(type="tool_use", id="tool-1", name="web_search", input={}))
                b = fresh([first, response("Final answer")])
                result = asyncio.run(b._generate_claude_response(1,
                    [{"role": "user", "content": "Find this."}], "Be helpful.", thinking=thinking, effort=effort))
                self.assertEqual(result[0], "Final answer")
                calls = b.claude_client.messages.create.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0].kwargs["thinking"]["type"], "adaptive")
                self.assertEqual(calls[0].kwargs["extra_body"]["output_config"]["effort"], effort or "low")
                # And it went over the right transport: only the 64k tier may
                # stream, and when it does BOTH requests must (the tool-loop
                # continuation reuses the same kwargs).
                self.assertEqual(len(b.claude_client.messages.streamed),
                                 2 if effort in ("xhigh", "max") else 0)
                self.assertIs(calls[1].kwargs["messages"][-2]["content"], first.content)
                self.assertEqual(b.claude_provider.total_requests, 1)
                self.assertEqual(b.claude_provider.total_output_tokens, 100)

    def test_search_also_uses_adaptive_low(self):
        b = fresh([response("Search answer")])
        asyncio.run(b._web_search("query", NS(), 1))
        request = b.claude_client.messages.create.call_args.kwargs
        self.assertEqual(request["thinking"]["type"], "adaptive")
        self.assertEqual(request["extra_body"]["output_config"]["effort"], "low")

    def test_qwen_rate_matches_its_backend(self):
        if module.QWEN_PROVIDER.backend == "alibaba":
            self.assertEqual(module.QWEN_PROVIDER.cached_input_cost_per_million, .014)
        else:
            self.assertTrue(module.QWEN_PROVIDER.model_id.startswith("accounts/fireworks/models/qwen"))
            self.assertEqual(module.QWEN_PROVIDER.cached_input_cost_per_million, .25)


if __name__ == "__main__":
    unittest.main()
