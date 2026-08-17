"""Small helper for the documented Graphify update workflow."""
from pathlib import Path
import subprocess


def update_graph(project_root: Path) -> int:
    """Incrementally update Graphify output when the graphify CLI is installed."""
    command = ["graphify", str(project_root), "--update", "--no-viz"]
    try:
        return subprocess.run(command, cwd=project_root, check=False).returncode
    except FileNotFoundError:
        return 127
