def read_yaml(path: str) -> dict[str, any]:
    import yaml

    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Yaml loading failed due to: {e}")
