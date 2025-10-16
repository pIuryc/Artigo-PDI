import kagglehub
import shutil
from pathlib import Path

print("🔹 Baixando dataset do Kaggle...")
path = kagglehub.dataset_download("ma7555/cat-breeds-dataset")
print(f"✅ Dataset baixado em: {path}")

dest = Path("data/raw")
dest.mkdir(parents=True, exist_ok=True)

for item in Path(path).iterdir():
    target = dest / item.name
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)

print(f"✅ Copiado para: {dest.resolve()}")
