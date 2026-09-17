"""Build with local Docker, push the image, and deploy to AgentKit Runtime."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess

import yaml
from dotenv import load_dotenv


def prepare_source(directory):
    """Bundle the pinned repository commit before Docker starts."""
    commit = re.search(
        r"/archive/([0-9a-f]{40})\.zip",
        (directory / "requirements.txt").read_text(),
    ).group(1)
    repository = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=directory, text=True
    ).strip()
    vendor = directory / "vendor"
    vendor.mkdir(exist_ok=True)
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--output={vendor / 'veadk.zip'}",
            commit,
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "veadk",
        ],
        cwd=repository,
        check=True,
    )
    print(f"Bundled VeADK commit {commit} from local Git.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="skill-code-orchestrator")
    parser.add_argument("--project", default="default")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    load_dotenv(directory / ".env")
    region = os.getenv("REGION", "cn-beijing")
    keys = (
        "AGENTKIT_SKILL_TOOL_ID",
        "AGENTKIT_CODEENV_TOOL_ID",
        "REGION",
        "MODEL_AGENT_NAME",
        "MODEL_AGENT_API_KEY",
        "MODEL_AGENT_API_BASE",
        "MODEL_AGENT_PROVIDER",
        "VOLCENGINE_ACCESS_KEY",
        "VOLCENGINE_SECRET_KEY",
        "VOLCENGINE_SESSION_TOKEN",
        "AGENTKIT_TOOL_SERVICE_CODE",
        "AGENTKIT_TOOL_HOST",
        "AGENTKIT_TOOL_SCHEME",
    )
    path = directory / "agentkit.yaml"
    # Preserve build artifacts and runtime identity on subsequent deployments.
    config = (
        yaml.safe_load(path.read_text())
        if path.exists()
        else {
            "common": {},
            "launch_types": {
                "hybrid": {
                    "cr_instance_name": "Auto",
                    "cr_namespace_name": "agentkit",
                    "runtime_role_name": "Auto",
                    "runtime_apikey_name": "Auto",
                    "runtime_auth_type": "key_auth",
                    "runtime_network": {"mode": "public"},
                }
            },
            "docker_build": {"platform": "auto"},
        }
    )
    envs = dict(config["common"].get("runtime_envs", {}))
    envs.update({key: os.environ[key] for key in keys if os.environ.get(key)})
    region = os.getenv("REGION", envs.get("REGION", region))
    envs["REGION"] = region
    for key in ("AGENTKIT_SKILL_TOOL_ID", "AGENTKIT_CODEENV_TOOL_ID"):
        if not envs.get(key):
            parser.error(f"Set {key} in the environment or this case's .env")
    strategies = config["launch_types"]
    if "hybrid" not in strategies:
        # Reuse registry/runtime identity; cloud pipeline fields are not used.
        strategies["hybrid"] = {
            key: value
            for key, value in strategies.get("cloud", {}).items()
            if not key.startswith(("cp_", "tos_")) and key != "build_timeout"
        }
    if (
        strategies["hybrid"].get("cr_instance_name")
        == "agentkit-platform-{{account_id}}"
    ):
        strategies["hybrid"]["cr_instance_name"] = "Auto"
    config["common"].update(
        {
            "agent_name": args.name,
            "entry_point": "app.py",
            "description": "Root coordinator with Skill and CodeEnv remote agents",
            "language": "Python",
            "language_version": "3.12",
            "agent_type": "Basic App",
            "dependencies_file": "requirements.txt",
            "launch_type": "hybrid",
            "cloud_provider": "volcengine",
            "runtime_envs": envs,
        }
    )
    config["launch_types"]["hybrid"].update(
        {
            "region": region,
            "project_name": args.project,
            "runtime_name": args.name,
            "image_tag": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        }
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(fd, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    print(
        "Prepared agentkit.yaml: local Docker build → registry push → cloud Runtime.",
        flush=True,
    )
    prepare_source(directory)
    if not args.prepare_only:
        subprocess.run(
            ["ak", "launch", "--config-file", str(path), "--platform", "linux/amd64"],
            cwd=directory,
            env={**envs, **os.environ},
            check=True,
        )


if __name__ == "__main__":
    main()
