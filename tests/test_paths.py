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


def test_explicit_workspace_root_is_canonical_from_nested_directory(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    nested = root / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.setenv("MEMORY_WORKSPACE_ROOT", str(root / "src" / ".."))

    assert paths_module._resolve_workspace_root() == root.resolve()


def test_symlinked_and_real_workspace_paths_share_identity(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    link = tmp_path / "repo-link"
    link.symlink_to(root, target_is_directory=True)

    assert paths_module._workspace_dir_name(root.resolve(), "", "") == paths_module._workspace_dir_name(
        link.resolve(), "", ""
    )


def test_shared_temp_root_marker_does_not_claim_unrelated_child(monkeypatch, tmp_path):
    child = tmp_path / "job" / "nested"
    child.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='ambient'\n", encoding="utf-8")
    monkeypatch.setattr(paths_module.tempfile, "gettempdir", lambda: str(tmp_path))

    assert paths_module._find_workspace_marker(child) is None


def test_workspace_discovery_stops_at_device_boundary(monkeypatch, tmp_path):
    mount = tmp_path / "mount"
    nested = mount / "repo" / "src"
    nested.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='outside'\n", encoding="utf-8")
    real_stat = paths_module.Path.stat

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == tmp_path:
            values = list(result)
            values[2] = result.st_dev + 1
            return type(result)(values)
        return result

    monkeypatch.setattr(paths_module.Path, "stat", fake_stat)

    assert paths_module._find_workspace_marker(nested) is None
