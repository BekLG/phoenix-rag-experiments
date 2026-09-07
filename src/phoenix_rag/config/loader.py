"""
config/loader.py
================
Reading and writing configuration. YAML on disk, :class:`AppConfig` in memory.

WHY YAML
--------
The config file is meant to be *tweaked by hand* -- it is the knob panel for the
whole optimizer -- and it is meant to be committed. JSON is poor at both: no
comments, and quoting every key makes a multi-line ``prompt_template`` painful to
read. YAML gives block scalars for the prompt and lets each tunable carry a note
saying what it does.

CREDENTIALS NEVER APPEAR HERE
-----------------------------
A provider block names the *environment variable* holding its key
(``api_key_env: OPENAI_API_KEY``), never the key itself. The value lives in
``.env``, which is gitignored. That separation is what makes the YAML safe to
commit, which is the entire point of having the tunables in a file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from phoenix_rag.config.schema import AppConfig
from phoenix_rag.workspace import Workspace, active_workspace

logger = logging.getLogger("phoenix_rag.config")

_env_loaded = False


class _ReadableDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line strings as block scalars.

    Without this, a saved ``prompt_template`` comes back as one long
    double-quoted line full of ``\\n`` escapes. The prompt is the field an
    operator is most likely to hand-edit, so it has to stay readable across a
    load/save round trip.
    """


def _represent_str(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ReadableDumper.add_representer(str, _represent_str)


def load_env(dotenv_path: str | Path | None = None) -> None:
    """Load ``.env`` into the process environment, at most once.

    Config objects reference credentials by variable name and read them at use
    time, so this has to have run before any provider client is constructed.
    Entry points call it; library code does not, because silently reading a
    ``.env`` from the caller's working directory is a side effect a library
    should not impose on import.

    A missing python-dotenv or missing file is not an error -- the variables may
    simply already be exported.
    """
    global _env_loaded
    if _env_loaded and dotenv_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        logger.debug("python-dotenv not installed; relying on exported variables")
        _env_loaded = True
        return
    load_dotenv(dotenv_path) if dotenv_path else load_dotenv()
    _env_loaded = True


# --------------------------------------------------------------------------
# Legacy migration
# --------------------------------------------------------------------------

def _migrate_legacy(data: dict) -> dict:
    """Translate a pre-provider config dict into the current shape.

    The old format had a single flat ``mistral`` block carrying one API key and
    four model names. Those four names map exactly onto the four roles, which is
    what made per-role providers a natural extension rather than a redesign, so
    the translation is mechanical.
    """
    if "providers" in data or "mistral" not in data:
        return data

    legacy = data.pop("mistral") or {}
    pacing = {
        key: legacy[key]
        for key in ("requests_per_minute", "max_retries", "base_backoff_seconds")
        if key in legacy
    }

    def block(model_key: str, default_model: str) -> dict:
        return {
            "backend": "mistral",
            "model": legacy.get(model_key, default_model),
            "api_key_env": "MISTRAL_API_KEY",
            **pacing,
        }

    data["providers"] = {
        "embedding": block("embedding_model", "mistral-embed"),
        "generation": block("generation_model", "mistral-small-latest"),
        "optimizer": block("optimizer_model", "mistral-large-latest"),
        "judge": block("judge_model", "mistral-large-latest"),
    }
    # An inline api_key in a committed config is exactly what api_key_env exists
    # to prevent; say so rather than quietly dropping it.
    if legacy.get("api_key"):
        logger.warning(
            "Legacy config contained an inline mistral.api_key. It has NOT been "
            "carried over -- put the value in .env as MISTRAL_API_KEY instead, and "
            "consider rotating it if this file was ever committed."
        )
    return data


# --------------------------------------------------------------------------
# Load / save
# --------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML (or legacy JSON) file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # yaml.safe_load parses JSON too, since JSON is a YAML subset -- so a
    # leftover .json config loads through the same path.
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )

    return AppConfig.from_dict(_migrate_legacy(data))


def save_config(config: AppConfig, path: str | Path) -> Path:
    """Write `config` to `path` as YAML. Creates the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            config.to_dict(),
            Dumper=_ReadableDumper,
            sort_keys=False,        # declaration order is meaningful to a reader
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )
    return path


def load_or_create_default_config(
    workspace: Workspace | None = None,
) -> AppConfig:
    """The workspace's config, created from defaults if absent.

    Resolution order:
      1. ``<workspace>/config/config.yaml`` if it exists.
      2. ``<workspace>/config/default_config.json`` (the pre-YAML location) if
         it exists -- migrated to YAML and rewritten, so this happens once.
      3. Defaults, written out as YAML so there is a file to edit.

    Calls :func:`load_env` first, because every caller of this function goes on
    to construct provider clients that need the credentials present.
    """
    load_env()
    workspace = workspace or active_workspace()

    if workspace.config_path.exists():
        return load_config(workspace.config_path)

    if workspace.legacy_config_path.exists():
        logger.info(
            "Migrating %s -> %s (JSON config format is superseded by YAML)",
            workspace.legacy_config_path,
            workspace.config_path,
        )
        config = load_config(workspace.legacy_config_path)
        save_config(config, workspace.config_path)
        return config

    config = AppConfig()
    save_config(config, workspace.config_path)
    logger.info("Wrote a default configuration to %s", workspace.config_path)
    return config
