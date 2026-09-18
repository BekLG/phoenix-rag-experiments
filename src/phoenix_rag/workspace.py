"""
workspace.py
============
Where Phoenix RAG reads and writes on disk.

WHY THIS EXISTS
---------------
These paths used to be module-level constants in config.py, derived from the
source tree and created on import:

    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    ...
    for _dir in (DATA_DIR, RESULTS_DIR, ...):
        _dir.mkdir(parents=True, exist_ok=True)

Both halves of that are fine in a script and wrong in an installed library:

  1. ``Path(__file__).parent`` is the *package's own* directory. Once this is
     pip-installed, that is site-packages -- so an optimization run would write
     its results, logs and FAISS indexes into the user's virtualenv, where they
     are invisible, unbackuped, and destroyed by the next reinstall.
  2. The mkdir loop ran at import time, so ``import phoenix_rag`` created five
     directories as a side effect, in whatever directory the process happened
     to start in, before the caller had asked for anything.

A Workspace makes the location explicit and the creation deliberate. Nothing in
this module touches the filesystem until someone calls :meth:`Workspace.ensure`.

The default root is the current working directory, so running from a checkout
resolves to the same ``data/``, ``results/`` and ``generated_questions/`` the
scripts always used -- existing runs keep their artifacts. Set
``PHOENIX_RAG_WORKSPACE`` to put them somewhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ENV_VAR = "PHOENIX_RAG_WORKSPACE"


@dataclass(frozen=True)
class Workspace:
    """A root directory plus the subdirectories Phoenix RAG uses under it.

    Frozen because a workspace is an address, not state: code that wants a
    different location builds a different Workspace rather than mutating a
    shared one out from under whoever else is holding it.
    """

    root: Path

    def __post_init__(self) -> None:
        # Normalize once, here, so every derived path is absolute regardless of
        # what the caller passed and of any later os.chdir.
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    # -- construction -----------------------------------------------------

    @classmethod
    def default(cls) -> "Workspace":
        """The workspace to use when the caller has not named one.

        ``PHOENIX_RAG_WORKSPACE`` if set, else the current working directory.
        Deliberately *not* derived from ``__file__`` -- see module docstring.
        """
        configured = os.getenv(WORKSPACE_ENV_VAR)
        return cls(Path(configured) if configured else Path.cwd())

    # -- derived locations ------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def generated_questions_dir(self) -> Path:
        return self.root / "generated_questions"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def corpus_dir(self) -> Path:
        """Multi-document corpus root: the manifest plus one FAISS index
        directory per (embedding_model, chunk_size, chunk_overlap) variant.

        Not created by :meth:`ensure` -- corpus.py makes it on first use, so
        single-document runs leave no empty corpus directory behind.
        """
        return self.data_dir / "corpus"

    @property
    def config_path(self) -> Path:
        """The YAML config this workspace loads by default."""
        return self.config_dir / "config.yaml"

    @property
    def legacy_config_path(self) -> Path:
        """The pre-YAML config location, still read once for migration."""
        return self.config_dir / "default_config.json"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "phoenix_rag.log"

    # -- the only filesystem mutation in this module -----------------------

    def ensure(self) -> "Workspace":
        """Create the standard subdirectories. Idempotent; returns self.

        Call this from an entry point (CLI, notebook, front-end), not from
        library code at import time.
        """
        for directory in (
            self.data_dir,
            self.results_dir,
            self.generated_questions_dir,
            self.logs_dir,
            self.config_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.root)


# --------------------------------------------------------------------------
# Process-wide active workspace
# --------------------------------------------------------------------------
# AppConfig's path defaults need *a* workspace when the caller has not supplied
# one (bare `AppConfig()` is a documented, tested construction). This holds the
# one the process is using. Resolved lazily on first read so that merely
# importing phoenix_rag still reads no environment and touches no disk.

_active: Workspace | None = None


def active_workspace() -> Workspace:
    """The workspace this process is using, resolving the default on first call."""
    global _active
    if _active is None:
        _active = Workspace.default()
    return _active


def use_workspace(workspace: Workspace | str | Path) -> Workspace:
    """Set the active workspace and return it.

    Process-wide by design, the same way storage.configure_results_dir is: it
    is a redirect, not a scope. Entry points call this once, early.
    """
    global _active
    _active = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return _active


def reset_workspace() -> None:
    """Forget the active workspace so the next read re-resolves the default.

    Exists for tests, which need to point the library at a tmpdir and then put
    it back without leaking state into the next test.
    """
    global _active
    _active = None
