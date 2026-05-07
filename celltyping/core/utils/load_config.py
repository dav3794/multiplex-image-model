from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.get("base_path")
    if base_path is None:
        raise ValueError("YAML must define 'base_path'.")

    base_path = str(Path(base_path))

    def replace_base(obj):
        if isinstance(obj, dict):
            return {k: replace_base(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_base(v) for v in obj]
        elif isinstance(obj, str):
            if obj.startswith("/.../"):
                # replace the /.../ prefix with base_path
                return base_path + obj[4:]
            return obj
        else:
            return obj

    cfg = replace_base(cfg)
    return cfg
