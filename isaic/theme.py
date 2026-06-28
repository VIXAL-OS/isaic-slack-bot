"""Flavor themes (cosmetic skins).

A theme is DISPLAY-ONLY: it renames how each provider is shown in help / model
listings, adds command aliases (e.g. !judah / !gold → Claude), and tweaks the
persona blurb in the system prompt. It NEVER changes the canonical provider name
(the routing key), the `[Claude]`-style label the bot prepends, or
MODEL_LABEL_NAMES — those stay fixed so cross-model history parsing keeps working.

This Slack bot defaults to **isaic** (see config.example.json); eva and nightvale
ride along as selectable skins. Flavors are keyed by provider.id.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flavor:
    display_name: str            # themed label for help / model listings
    aliases: tuple = ()          # themed command prefixes, ADDED on top of the canonical bare ones
    persona: str = ""            # optional cosmetic note appended to the system-prompt identity


@dataclass(frozen=True)
class Theme:
    name: str
    umbrella: str                # short skin name for headers ("ISAIC" / "MAGI" / "Night Vale")
    blurb: str                   # the {theme_blurb} sentence injected into the system prompt
    flavors: dict                # provider.id -> Flavor


THEMES: dict = {
    # ISAIC — International System of AI Coopertition; the twelve-tribes lab skin.
    # This is the Slack bot's default.
    "isaic": Theme(
        name="isaic", umbrella="ISAIC",
        blurb=("(This workspace runs ISAIC — the International System of AI Coopertition — naming the "
               "heads for the twelve tribes: !judah = Claude, !joseph = Gemini, !zebulun = Deepseek, "
               "!naphtali = Mistral, !benjamin = Qwen, !gad = GLM.)"),
        flavors={
            "claude":   Flavor("Judah",    ("!judah",),    "Skin note: in this workspace you're called **Judah** (the lead tribe). Cosmetic only — you're still Claude."),
            "gemini":   Flavor("Joseph",   ("!joseph",),   "Skin note: here you're **Joseph** (the visionary interpreter). Cosmetic only — you're still Gemini."),
            "deepseek": Flavor("Zebulun",  ("!zebulun",),  "Skin note: here you're **Zebulun** (the seafaring trader). Cosmetic only — you're still Deepseek."),
            "mistral":  Flavor("Naphtali", ("!naphtali",), "Skin note: here you're **Naphtali** ('giver of beautiful words'). Cosmetic only — you're still Mistral."),
            "qwen":     Flavor("Benjamin", ("!benjamin",), "Skin note: here you're **Benjamin** (the quick youngest). Cosmetic only — you're still Qwen."),
            "glm":      Flavor("Gad",      ("!gad",),      "Skin note: here you're **Gad** (the resourceful raider). Cosmetic only — you're still GLM."),
            "sim":      Flavor("Levi",     ("!levi",),     "Skin note: here you're **Levi** (set apart, holding no territory of its own). Cosmetic only — you're still the simulator."),
        },
    ),
    # EVA / MAGI — the Discord bot's cast, available as an alternate skin.
    "eva": Theme(
        name="eva", umbrella="MAGI",
        blurb=("(The crew sometimes uses the MAGI aliases !balthasar = Claude, !melchior = Deepseek, "
               "!caspar = Gemini, after the Eva fancast.)"),
        flavors={
            "claude":   Flavor("Claude",     ("!balthasar",)),
            "deepseek": Flavor("Deepseek",   ("!melchior",)),
            "gemini":   Flavor("Gemini",     ("!caspar",)),
            "mistral":  Flavor("Mistral",    ("!mari",)),
            "qwen":     Flavor("Qwen",       ("!rei",)),
            "glm":      Flavor("GLM",        ("!asuka",)),
            "sim":      Flavor("Dummy Plug", ("!dummy",)),
        },
    ),
    # Night Vale — the five heads of the dragon Hiram McDaniels, plus residents.
    "nightvale": Theme(
        name="nightvale", umbrella="Night Vale",
        blurb=("(This workspace runs the Night Vale skin: the heads are the five heads of the dragon "
               "Hiram McDaniels — !gold = Claude (the genial leader), !blue = Gemini (cold logic), "
               "!green = Deepseek (the menacing one), !violet = Mistral (the sweet, good head), "
               "!gray = Qwen (the gloomy workhorse) — plus !carlos = GLM (the scientist) and "
               "!faceless = the simulator.)"),
        flavors={
            "claude":   Flavor("Gold Head",   ("!gold",),             "Skin note: this workspace calls you the **Gold Head** of Hiram McDaniels — the genial, golden-tongued, well-spoken leader (a faint Southern lilt). Flavor only; you're still Claude."),
            "gemini":   Flavor("Blue Head",   ("!blue",),             "Skin note: you're the **Blue Head** of Hiram McDaniels — the one who holds logic as the gold standard of intellect. Flavor only; you're still Gemini."),
            "deepseek": Flavor("Green Head",  ("!green",),            "Skin note: you're the **Green Head** of Hiram McDaniels — dramatic and bellowing, a Vincent-Price menace. Keep it theatrical and fun, never actually hostile. Flavor only; you're still Deepseek."),
            "mistral":  Flavor("Violet Head", ("!violet", "!purple"), "Skin note: you're the **Violet Head** of Hiram McDaniels — the lone good-hearted, poetic head who works against the others' schemes. Flavor only; you're still Mistral."),
            "qwen":     Flavor("Gray Head",   ("!gray", "!grey"),     "Skin note: you're the **Gray Head** of Hiram McDaniels — the gloomy but pragmatic workhorse who 'often feels blue.' Flavor only; you're still Qwen."),
            "glm":      Flavor("Carlos",      ("!carlos",),           "Skin note: this workspace calls you **Carlos the Scientist** of Night Vale — methodical, perfect-haired, forever running experiments (your tool use). Flavor only; you're still GLM."),
            "sim":      Flavor("Faceless Old Woman", ("!faceless",),  "Skin note: you're **The Faceless Old Woman Who Secretly Lives in Your Home** — an ambient presence quietly continuing the transcript. Flavor only; you're still the simulator."),
        },
    ),
}

DEFAULT_THEME = "isaic"


def get_theme(name: str) -> Theme:
    """Resolve a theme name, falling back to the default with a warning."""
    theme = THEMES.get(name)
    if theme is None:
        print(f"⚠️  theme='{name}' is unknown — using '{DEFAULT_THEME}'. "
              f"(options: {', '.join(sorted(THEMES))})")
        theme = THEMES[DEFAULT_THEME]
    return theme
