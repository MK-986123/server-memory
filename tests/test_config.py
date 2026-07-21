"""Tests for cross-platform path and auth token configuration defaults."""

from pathlib import Path

from server_memory import config as config_module


def test_memory_config_uses_platform_defaults(monkeypatch):
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    monkeypatch.delenv("MEMORY_AUTH_TOKEN_PATH", raising=False)
    monkeypatch.delenv("MEMORY_GLOBAL_DB_PATH", raising=False)
    monkeypatch.setattr(config_module, "default_db_path", lambda: Path("/tmp/app-data/memory.db"))
    monkeypatch.setattr(
        config_module,
        "default_auth_token_path",
        lambda: Path("/tmp/app-runtime/auth.token"),
    )
    monkeypatch.setattr(
        config_module,
        "default_global_db_path",
        lambda: Path("/tmp/app-data/global/preferences.db"),
    )

    cfg = config_module.MemoryConfig()

    assert cfg.db_path == Path("/tmp/app-data/memory.db")
    assert cfg.auth_token_path == Path("/tmp/app-runtime/auth.token")
    assert cfg.global_db_path == Path("/tmp/app-data/global/preferences.db")


def test_memory_config_respects_explicit_path_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_DB_PATH", "/custom/db.sqlite")
    monkeypatch.setenv("MEMORY_AUTH_TOKEN_PATH", "/custom/auth.token")
    monkeypatch.setenv("MEMORY_GLOBAL_DB_PATH", "/custom/global.sqlite")

    cfg = config_module.MemoryConfig()

    assert cfg.db_path == Path("/custom/db.sqlite")
    assert cfg.auth_token_path == Path("/custom/auth.token")
    assert cfg.global_db_path == Path("/custom/global.sqlite")


def test_memory_config_strips_surrounding_whitespace_on_explicit_paths(monkeypatch):
    monkeypatch.setenv("MEMORY_DB_PATH", "  /custom/db.sqlite  ")
    monkeypatch.setenv("MEMORY_AUTH_TOKEN_PATH", "\t/custom/auth.token\n")
    monkeypatch.setenv("MEMORY_GLOBAL_DB_PATH", " /custom/global.sqlite ")

    cfg = config_module.MemoryConfig()

    assert cfg.db_path == Path("/custom/db.sqlite")
    assert cfg.auth_token_path == Path("/custom/auth.token")
    assert cfg.global_db_path == Path("/custom/global.sqlite")


def test_memory_config_treats_blank_path_overrides_as_unset(monkeypatch):
    monkeypatch.setattr(config_module, "default_db_path", lambda: Path("/tmp/default/memory.db"))
    monkeypatch.setattr(
        config_module,
        "default_auth_token_path",
        lambda: Path("/tmp/default/auth.token"),
    )
    monkeypatch.setattr(
        config_module,
        "default_global_db_path",
        lambda: Path("/tmp/default/global.db"),
    )

    for value in ("", "   ", "\t\n"):
        monkeypatch.setenv("MEMORY_DB_PATH", value)
        monkeypatch.setenv("MEMORY_AUTH_TOKEN_PATH", value)
        monkeypatch.setenv("MEMORY_GLOBAL_DB_PATH", value)

        cfg = config_module.MemoryConfig()

        assert cfg.db_path == Path("/tmp/default/memory.db")
        assert cfg.auth_token_path == Path("/tmp/default/auth.token")
        assert cfg.global_db_path == Path("/tmp/default/global.db")
        # Blank overrides must never collapse to cwd / repo root.
        assert cfg.db_path != Path(".")
        assert cfg.db_path != Path.cwd()
        assert cfg.global_db_path != Path(".")
        assert cfg.global_db_path != Path.cwd()
        assert cfg.auth_token_path != Path(".")
        assert cfg.auth_token_path != Path.cwd()


def test_env_path_or_default_helper_matches_contract(monkeypatch, tmp_path):
    default = tmp_path / "platform" / "memory.db"
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    assert config_module.env_path_or_default("MEMORY_DB_PATH", default) == default

    monkeypatch.setenv("MEMORY_DB_PATH", "")
    assert config_module.env_path_or_default("MEMORY_DB_PATH", default) == default

    monkeypatch.setenv("MEMORY_DB_PATH", "   ")
    assert config_module.env_path_or_default("MEMORY_DB_PATH", default) == default

    explicit = tmp_path / "explicit.db"
    monkeypatch.setenv("MEMORY_DB_PATH", f"  {explicit}  ")
    assert config_module.env_path_or_default("MEMORY_DB_PATH", default) == explicit


def test_memory_config_ensures_auth_token_dir(tmp_path):
    token_path = tmp_path / "runtime" / "auth.token"
    cfg = config_module.MemoryConfig(auth_token_path=token_path)

    cfg.ensure_auth_token_dir()

    assert token_path.parent.is_dir()


def test_memory_config_ensures_global_db_dir(tmp_path):
    global_db_path = tmp_path / "global" / "preferences.db"
    cfg = config_module.MemoryConfig(global_db_path=global_db_path)

    cfg.ensure_global_db_dir()

    assert global_db_path.parent.is_dir()
