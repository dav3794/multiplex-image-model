import os
import sys

from ruamel.yaml import YAML

from .load_config import load_config


def load_yaml(path):
    yaml = YAML()
    with open(path, "r") as f:
        return yaml.load(f)


def add_model_repo(repo_path: str):
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.append(repo_path)
