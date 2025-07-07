from pathlib import Path

# Base directories
base_dirs = [
    "./models",
    "./datasets",
    "./prompts"
]

# __init__.py content for models to expose `create_model`
model_init_content = "from .encoders import create_model\n"

# Create __init__.py for each directory
for dir_path in base_dirs:
    init_file = Path("/mnt/data") / dir_path / "__init__.py"
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if "models" in dir_path:
        init_file.write_text(model_init_content, encoding="utf-8")
    else:
        init_file.write_text("", encoding="utf-8")

# Return paths of created files
[Path("/mnt/data") / d / "__init__.py" for d in base_dirs]