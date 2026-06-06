from typer.testing import CliRunner

from locomaniformer.cli import app


def test_hello_command() -> None:
    result = CliRunner().invoke(app, ["hello", "tester"])

    assert result.exit_code == 0
    assert "Hello, tester." in result.stdout
