"""Tests for workspace-aware path resolution."""

from server_memory import paths as paths_module


def test_default_db_path_uses_workspace_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paths_module,
        "app_paths",
        lambda: paths_module.AppPaths(
            data_dir=tmp_path / "data",
            config_dir=tmp_path / "config",
            runtime_dir=tmp_path / "runtime",
        ),
    )
    monkeypatch.setenv("MEMORY_WORKSPACE_ID", "Project X")
    monkeypatch.delenv("MEMORY_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)

    db_path = paths_module.default_db_path()

    assert db_path.parent.parent == tmp_path / "data" / "workspaces"
    assert db_path.name == "memory.db"
    assert db_path.parent.name.startswith("project-x_")
    assert len(db_path.parent.name.rsplit("_", 1)[1]) == 12


def test_default_db_path_discovers_workspace_root_from_markers(monkeypatch, tmp_path):
    repo_root = tmp_path / "example-repo"
    nested_dir = repo_root / "src" / "feature"
    nested_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")

    monkeypatch.setattr(
        paths_module,
        "app_paths",
        lambda: paths_module.AppPaths(
            data_dir=tmp_path / "data",
            config_dir=tmp_path / "config",
            runtime_dir=tmp_path / "runtime",
        ),
    )
    monkeypatch.delenv("MEMORY_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("MEMORY_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    monkeypatch.chdir(nested_dir)

    db_path = paths_module.default_db_path()

    assert db_path.parent.parent == tmp_path / "data" / "workspaces"
    assert db_path.parent.name.startswith("example-repo_")


def test_default_db_path_falls_back_to_global_when_no_workspace(monkeypatch, tmp_path):
    no_project_dir = tmp_path / "scratch"
    no_project_dir.mkdir()

    monkeypatch.setattr(
        paths_module,
        "app_paths",
        lambda: paths_module.AppPaths(
            data_dir=tmp_path / "data",
            config_dir=tmp_path / "config",
            runtime_dir=tmp_path / "runtime",
        ),
    )
    monkeypatch.delenv("MEMORY_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("MEMORY_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    monkeypatch.chdir(no_project_dir)

    db_path = paths_module.default_db_path()

    assert db_path == tmp_path / "data" / "memory.db"
