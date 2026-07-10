"""Offline checks for the !research panel fixes (no API keys, no Slack).

Covers:
  1. _looks_like_tool_call_dump  — catches the DeepSeek content-serialized call,
     and (critically) does NOT fire on legitimate prose about JSON / web_search.
  2. _tool_call_extra_content    — reads Gemini's thought_signature off a typed
     attribute, off model_extra, and returns None when absent.
  3. The tool-loop rebuild        — echoes extra_content back per call.
"""
import sys, types, json, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaic import core

B = core.IsaicBot
ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------- dump detector
# Bind the unbound method to a stub carrying only what it reads.
stub = types.SimpleNamespace(
    OPENAI_COMPATIBLE_TOOLS=B.OPENAI_COMPATIBLE_TOOLS,
    _TOOL_TOKEN_RE=B._TOOL_TOKEN_RE,
)
dump = lambda t: B._looks_like_tool_call_dump(stub, t)

print("_looks_like_tool_call_dump — SHOULD FIRE (true positives)")
# The exact shape from deepseek-ai/DeepSeek-V3#1244: prose, then name+JSON.
check("deepseek #1244 shape", dump('好的，我来搜索。web_search{"query": "EEG emotion datasets"}'), True)
check("bare name+json", dump('web_search{"query": "x"}'), True)
check("name space json", dump('web_search {"query": "x"}'), True)
check("functions. prefix", dump('functions.web_search{"query": "x"}'), True)
check("template token U+2581", dump("<|tool▁calls▁begin|><|tool▁call▁begin|>function"), True)
check("template token ascii", dump("<|tool_calls_begin|>web_search"), True)
check("template token sep", dump("blah <|tool▁sep|> blah"), True)
check("whole msg is tool-call obj", dump('{"name": "web_search", "arguments": {"query": "x"}}'), True)
check("whole msg tool_call key", dump('{"tool_call": {"name": "web_search"}}'), True)

print("_looks_like_tool_call_dump — MUST NOT FIRE (false-positive guards)")
# This bot answers questions ABOUT the web_search tool and about JSON, constantly.
check("prose naming the tool", dump("Use the web_search tool when you need fresh facts."), False)
check("prose w/ backtick tool", dump("I called `web_search` and got 5 hits."), False)
check("tool name then period", dump("The tool is named web_search. It takes a query."), False)
check("json in a fence", dump('Here is the schema:\n```json\n{"name": "web_search"}\n```\nUse it.'), False)
check("real answer w/ json obj", dump('The config is {"model": "gemini-3.1-pro"} — set it in config.json.'), False)
check("plain research answer", dump("DEAP is the best starting point. Its stimuli were music videos."), False)
check("empty", dump(""), False)
check("bare non-tool json", dump('{"query": "x"}'), False)
check("markdown w/ pipes", dump("| col | col |\n|---|---|"), False)
# A code sample that *shows* a call — has prose around it, no name{" adjacency.
check("code sample w/ kwargs", dump('Call it as web_search(query="x") in Python.'), False)

# ------------------------------------------------------- extra_content extractor
print("_tool_call_extra_content")
SIG = {"google": {"thought_signature": "CvcQAdHN2OekY10ClPFkYA=="}}


class TypedAttr:
    """openai SDK pydantic model with extra='allow' → extras become attributes."""
    extra_content = SIG
    def model_dump(self): return {"id": "1", "extra_content": SIG}


class DumpOnly:
    """Defensive path: extras only visible via model_dump()."""
    def model_dump(self): return {"id": "1", "extra_content": SIG}


class Nested:
    """extra_content itself parsed into a pydantic submodel."""
    class _E:
        def model_dump(self): return SIG
    extra_content = _E()
    def model_dump(self): return {"id": "1"}


class NoSig:
    def model_dump(self): return {"id": "1"}


check("typed attribute", B._tool_call_extra_content(TypedAttr()), SIG)
check("model_dump fallback", B._tool_call_extra_content(DumpOnly()), SIG)
check("nested pydantic", B._tool_call_extra_content(Nested()), SIG)
check("absent -> None", B._tool_call_extra_content(NoSig()), None)
check("empty dict -> None", B._tool_call_extra_content(types.SimpleNamespace(extra_content={})), None)
check("plain object, no attrs", B._tool_call_extra_content(object()), None)

# ------------------------------------------------- the rebuild echoes it back
print("tool-loop rebuild round-trips extra_content")
fn = types.SimpleNamespace(name="web_search", arguments='{"query":"x"}')
tc_sig = types.SimpleNamespace(id="call_1", function=fn, extra_content=SIG)
tc_bare = types.SimpleNamespace(id="call_2", function=fn)

rebuilt = []
for tc in (tc_sig, tc_bare):
    call = {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
    extra = B._tool_call_extra_content(tc)
    if extra:
        call["extra_content"] = extra
    rebuilt.append(call)

check("signed call carries sig", rebuilt[0].get("extra_content"), SIG)
check("unsigned call has no key", "extra_content" in rebuilt[1], False)
check("key path is google.thought_signature",
      rebuilt[0]["extra_content"]["google"]["thought_signature"], "CvcQAdHN2OekY10ClPFkYA==")
# _strip_internal_keys is a shallow top-level filter — must not eat extra_content.
stripped = B._strip_internal_keys([{"role": "assistant", "tool_calls": rebuilt, "_msg_id": 7}])
check("_strip_internal_keys preserves it",
      stripped[0]["tool_calls"][0]["extra_content"], SIG)
check("_strip_internal_keys drops _msg_id", "_msg_id" in stripped[0], False)
# And the whole message must survive json serialization (it goes out over HTTP).
check("serializes", json.loads(json.dumps(stripped[0]))["tool_calls"][0]["extra_content"], SIG)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
