"""Launcher regressions; run with python -m unittest meco.runtime.test_system."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

from .system import resolve_path


class InterpreterPathTests(unittest.TestCase):
    def test_symlinked_venv_keeps_its_environment(self):
        with tempfile.TemporaryDirectory(prefix="meco-venv-") as directory:
            environment = Path(directory) / "env"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            executable = environment / "bin/python"
            self.assertTrue(executable.is_symlink())
            configured = os.path.relpath(executable)
            command = [str(resolve_path(configured, preserve_symlinks=True)), "-c",
                       "import json, sys; print(json.dumps(sys.prefix))"]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(Path(json.loads(result.stdout)).resolve(), environment.resolve())
            self.assertEqual(resolve_path(configured), executable.resolve())

    def test_missing_interpreter_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="meco-path-") as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_path(str(Path(directory) / "missing"), preserve_symlinks=True)

    def test_unsafe_alias_or_target_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="meco-path-") as directory:
            root = Path(directory)
            alias = root / "bad alias"
            alias.symlink_to(sys.executable)
            target = root / "bad@target"
            target.touch()
            safe_alias = root / "safe-alias"
            safe_alias.symlink_to(target)
            for executable in (alias, safe_alias):
                with self.subTest(executable=executable):
                    with self.assertRaises(ValueError):
                        resolve_path(str(executable), preserve_symlinks=True)


if __name__ == "__main__":
    unittest.main()
