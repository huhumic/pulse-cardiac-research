"""Clone a private research repo from Kaggle using a read-only Kaggle Secret.

Do not print, persist or paste tokens in a notebook or remote URL. The token is
injected into a subprocess-only environment as an ephemeral Git config value.
"""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

REPO = "https://github.com/huhumic/pulse-cardiac-research.git"
DEST = Path("/kaggle/working/pulse-cardiac-research")


def main() -> None:
    if (DEST / "README.md").exists():
        print("Repository already present:", DEST)
        return
    env = os.environ.copy()
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_READ_TOKEN")
    except Exception as exc:
        token = None
        print("Kaggle GITHUB_READ_TOKEN secret not available; trying anonymous public clone.")
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {auth}"
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(DEST)],
                   check=True, env=env)
    print("Cloned:", DEST)


if __name__ == "__main__":
    main()
