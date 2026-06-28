"""ISAIC entrypoint — wire the core bot to the Slack adapter and run.

Usage:
    python main.py            # connect to Slack (Socket Mode) and run
    python main.py --check    # build objects + validate config, do NOT connect
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

MODEL_KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
              "FIREWORKS_API_KEY", "MISTRAL_API_KEY")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="ISAIC — Slack multi-model bot")
    ap.add_argument("--check", action="store_true",
                    help="build the bot + validate config without connecting to Slack")
    args = ap.parse_args()

    from isaic.core import IsaicBot
    bot = IsaicBot()  # reads config.json + .env, builds the provider registry

    if args.check:
        enabled = [p.name for p in bot.providers if p.enabled]
        ok = True
        try:
            import isaic.slack_adapter  # noqa: F401  (import-check the adapter)
            adapter_ok = "ok"
        except Exception as e:  # e.g. slack_bolt not installed
            adapter_ok = f"NOT importable ({type(e).__name__}: {e})"
            ok = False
        print(f"✓ config OK — theme={bot.theme.name}, "
              f"enabled heads={enabled or '(none — set an API key)'}")
        print(f"  slack_adapter import: {adapter_ok}")
        sys.exit(0 if ok else 1)

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        print("❌ SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required. See .env.example "
              "and create the app from slack-app-manifest.yaml.")
        sys.exit(1)
    if not any(os.getenv(k) for k in MODEL_KEYS):
        print(f"❌ Set at least one model API key ({', '.join(MODEL_KEYS)}). See .env.example.")
        sys.exit(1)

    from isaic.slack_adapter import SlackAdapter
    bot.attach_platform(SlackAdapter(bot_token, app_token))
    print("🚀 Starting ISAIC (Slack · Socket Mode)…")
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Shutting down.")


if __name__ == "__main__":
    main()
