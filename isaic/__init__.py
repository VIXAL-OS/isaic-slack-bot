"""ISAIC — International System of AI Coopertition.

A Slack-native, multi-model assistant: Claude (Judah) + Gemini (Joseph) +
Deepseek (Zebulun) with smart heuristic routing, plus open-weight heads
(Mistral/Naphtali, Qwen/Benjamin, GLM/Gad) and a base-model simulator (Levi).
Ported from the Discord "Hydra" bot via a ChatPlatform abstraction (Phase 4).

Kept import-light on purpose: `import isaic` pulls in no model/Slack SDKs. The
heavy objects live in `isaic.core` (IsaicBot) and `isaic.slack_adapter`.
"""

__version__ = "0.1.0"
