import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# --- CONFIGURAÇÕES ---
RAW_DIR = Path("data/raw")          # onde estão as imagens originais
OUT_DIR = Path("data/single")       # onde ficará a divisão
SPLIT_RATIO = (0.8, 0.1, 0.1)       # (train, val, test)
IMG_SIZE = (299, 299)               # tamanho padrão pro Xception
SEED = 42
random.seed(SEED)

# --- FUNÇÕES ---
def clear_or_create_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def resize_and_copy(src, dst):
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst)
    except Exception as e:
        print(f"[!] Erro ao processar {src}: {e}")

def split_data():
    print("🔹 Organizando dataset...")
    clear_or_create_dir(OUT_DIR)
    
    for cls in os.listdir(RAW_DIR):
        cls_path = RAW_DIR / cls
        if not cls_path.is_dir():
            continue
        
        images = [p for p in cls_path.glob("*.jpg")] + [p for p in cls_path.glob("*.jpeg")] + [p for p in cls_path.glob("*.png")]
        if not images:
            print(f"[!] Nenhuma imagem encontrada em {cls}")
            continue

        train, test = train_test_split(images, test_size=SPLIT_RATIO[1] + SPLIT_RATIO[2], random_state=SEED)
        val, test = train_test_split(test, test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]), random_state=SEED)

        for subset, data in zip(["train", "val", "test"], [train, val, test]):
            out_dir = OUT_DIR / subset / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in data:
                dst_path = out_dir / img_path.name
                resize_and_copy(img_path, dst_path)

    print("✅ Dataset organizado e redimensionado!")
    print(f"Diretório final: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    split_data()
