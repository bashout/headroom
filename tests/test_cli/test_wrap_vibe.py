"""Tests for `headroom wrap vibe` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_vibe_launch(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vibe launches with correct configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(
                    main, ["wrap", "vibe", "--port", "9000", "--", "--prompt", "test"]
                )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["tool_label"] == "VIBE"
    assert captured["agent_type"] == "vibe"
    assert captured["args"] == ("--prompt", "test")



def test_wrap_vibe_with_project_name(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project name is encoded in the URL when running from a project directory."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "vibe", "--port", "7000"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)

    providers: list[dict[str, Any]] = json.loads(env["VIBE_PROVIDERS"])
    assert providers[0]["api_base"] == "http://127.0.0.1:7000/p/my-project/v1"


def test_wrap_vibe_not_found(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error message when vibe binary is not found."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "vibe"])

    assert result.exit_code == 1
    assert "Error: 'vibe' not found in PATH" in result.output
    assert "Install Mistral Vibe: https://github.com/mistralai/mistral-vibe" in result.output


def test_wrap_vibe_custom_port(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom --port is passed to _launch_tool and appears in VIBE_PROVIDERS."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999
    env = captured["env"]
    providers: list[dict[str, Any]] = json.loads(env["VIBE_PROVIDERS"])
    assert providers[0]["api_base"] == "http://127.0.0.1:9999/v1"


def test_wrap_vibe_no_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-proxy flag prevents proxy startup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_vibe_code_graph(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--code-graph flag is passed to _launch_tool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe", "--code-graph"])

    assert result.exit_code == 0, result.output
    assert captured["code_graph"] is True


def test_wrap_vibe_learn_memory(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--learn and --memory flags are passed to _launch_tool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe", "--learn", "--memory"])

    assert result.exit_code == 0, result.output
    assert captured["learn"] is True
    assert captured["memory"] is True


def test_wrap_vibe_verbose(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--verbose flag is accepted by vibe command."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe", "--verbose"])

    assert result.exit_code == 0, result.output


def test_wrap_vibe_providers_json_structure(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIBE_PROVIDERS env var has correct JSON structure."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "vibe"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert "VIBE_PROVIDERS" in env

    providers: list[dict[str, Any]] = json.loads(env["VIBE_PROVIDERS"])
    assert isinstance(providers, list)
    assert len(providers) == 1
    assert providers[0]["name"] == "mistral"
    assert providers[0]["api_key_env_var"] == "MISTRAL_API_KEY"
    assert providers[0]["backend"] == "mistral"
    assert "api_base" in providers[0]
    assert providers[0]["browser_auth_base_url"] == "https://console.mistral.ai"
    assert (
        providers[0]["browser_auth_api_base_url"] == "https://console.mistral.ai/api"
    )


def test_wrap_vibe_no_context_tool(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool and --no-rtk flags are accepted and not passed to vibe."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                # Test --no-context-tool
                result = runner.invoke(main, ["wrap", "vibe", "--no-context-tool", "--", "test"])

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("test",)
    assert "--no-context-tool" not in captured["args"]

    captured.clear()
    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                # Test --no-rtk - should skip RTK setup
                with patch.object(wrap_mod, "_setup_vibe_rtk") as mock_setup:
                    result = runner.invoke(main, ["wrap", "vibe", "--no-rtk", "--", "test"])
                    mock_setup.assert_not_called()

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("test",)
    assert "--no-rtk" not in captured["args"]


def test_wrap_vibe_rtk_setup_called(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK setup is called when --no-rtk is not specified."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                with patch.object(wrap_mod, "_setup_vibe_rtk") as mock_setup:
                    result = runner.invoke(main, ["wrap", "vibe", "--", "test"])

    assert result.exit_code == 0, result.output
    mock_setup.assert_called_once()


def test_wrap_vibe_rtk_injection(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK instructions are injected into AGENTS.md."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    # Create a temporary home directory for vibe
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(
            wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file
        ):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=tmp_path / "rtk"):
                with patch.object(wrap_mod, "_launch_tool"):
                    with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                        result = runner.invoke(main, ["wrap", "vibe", "--", "test"])

    assert result.exit_code == 0, result.output
    assert agents_file.exists()
    content = agents_file.read_text()
    assert "<!-- headroom:rtk-instructions -->" in content
    assert "rtk git status" in content


def test_wrap_vibe_rtk_injection_skipped_with_no_rtk(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK instructions are NOT injected when --no-rtk is specified."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    # Create a temporary home directory for vibe
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"
    # Pre-create the file
    agents_file.write_text("# Existing content\n")

    with patch.object(wrap_mod.shutil, "which", return_value="vibe"):
        with patch.object(
            wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file
        ):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=tmp_path / "rtk"):
                with patch.object(wrap_mod, "_launch_tool"):
                    with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                        result = runner.invoke(main, ["wrap", "vibe", "--no-rtk", "--", "test"])

    assert result.exit_code == 0, result.output
    # The file should not have RTK instructions added
    content = agents_file.read_text()
    assert "<!-- headroom:rtk-instructions -->" not in content
    assert content == "# Existing content\n"


def test_wrap_vibe_prepare_only_with_rtk(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--prepare-only with default settings calls RTK setup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    with patch.object(
        wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file
    ):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=tmp_path / "rtk"):
            result = runner.invoke(main, ["wrap", "vibe", "--prepare-only"])

    assert result.exit_code == 0, result.output
    assert agents_file.exists()
    content = agents_file.read_text()
    assert "<!-- headroom:rtk-instructions -->" in content


def test_wrap_vibe_prepare_only_with_no_rtk(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--prepare-only with --no-rtk does not call RTK setup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_setup_vibe_rtk") as mock_setup:
        result = runner.invoke(main, ["wrap", "vibe", "--prepare-only", "--no-rtk"])

    assert result.exit_code == 0, result.output
    mock_setup.assert_not_called()


# =============================================================================
# Unwrap Vibe Tests
# =============================================================================


def test_unwrap_vibe_removes_rtk_instructions(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap vibe removes RTK instructions from AGENTS.md."""
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    # Create AGENTS.md with RTK instructions
    rtk_block = "<!-- headroom:rtk-instructions -->\n# RTK\nrtk git status\n<!-- /headroom:rtk-instructions -->"
    agents_file.write_text(f"# Some content\n\n{rtk_block}\n\n# More content\n")

    with patch.object(wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file):
        with patch.object(
            wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"
        ):
            result = runner.invoke(main, ["unwrap", "vibe", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "Removed rtk instructions from" in result.output

    # Verify the RTK block was removed but other content remains
    content = agents_file.read_text()
    assert "<!-- headroom:rtk-instructions -->" not in content
    assert "<!-- /headroom:rtk-instructions -->" not in content
    assert "# Some content" in content
    assert "# More content" in content


def test_unwrap_vibe_keeps_rtk_with_keep_flag(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap vibe with --keep-rtk does not remove RTK instructions."""
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    # Create AGENTS.md with RTK instructions
    rtk_block = "<!-- headroom:rtk-instructions -->\n# RTK\nrtk git status\n<!-- /headroom:rtk-instructions -->"
    agents_file.write_text(f"# Some content\n\n{rtk_block}\n")

    with patch.object(wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file):
        with patch.object(
            wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"
        ):
            result = runner.invoke(main, ["unwrap", "vibe", "--port", "9999", "--keep-rtk"])

    assert result.exit_code == 0, result.output
    assert "Kept rtk instructions in AGENTS.md" in result.output

    # Verify the RTK block was NOT removed
    content = agents_file.read_text()
    assert "<!-- headroom:rtk-instructions -->" in content


def test_unwrap_vibe_no_rtk_instructions_found(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap vibe reports when no RTK instructions are found."""
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    # Create AGENTS.md without RTK instructions
    agents_file.write_text("# Some content\n")

    with patch.object(wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file):
        with patch.object(
            wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"
        ):
            result = runner.invoke(main, ["unwrap", "vibe", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "No rtk instructions found in" in result.output


def test_unwrap_vibe_no_stop_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap vibe with --no-stop-proxy does not stop the proxy."""
    vibe_home = tmp_path / ".vibe"
    vibe_home.mkdir()
    agents_file = vibe_home / "AGENTS.md"

    # Create AGENTS.md with RTK instructions
    rtk_block = "<!-- headroom:rtk-instructions -->\n# RTK\n<!-- /headroom:rtk-instructions -->"
    agents_file.write_text(rtk_block)

    with patch.object(wrap_mod, "_get_vibe_agents_md_path", return_value=agents_file):
        with patch.object(
            wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"
        ) as mock_stop:
            result = runner.invoke(
                main, ["unwrap", "vibe", "--port", "9999", "--no-stop-proxy"]
            )

    assert result.exit_code == 0, result.output
    mock_stop.assert_not_called()
    assert "Stopped local Headroom proxy" not in result.output
