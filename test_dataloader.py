from pathlib import Path
import json

from src.config import Config, PROJECT_ROOT
from src.preprocessing import build_vocabulary
from src.data_loader import create_dataloader, load_labels

def main():
    log_dir = Config.RAW_LOG_DIR
    log_files = list(log_dir.glob("*.log"))
    print(f"Found {len(log_files)} log files in {log_dir}")

    # 1. 构建词表
    vocab = build_vocabulary(log_files, Config.MIN_TOKEN_FREQ)
    print(f"Vocab size: {len(vocab)}")

    # 2. 保存 vocab
    processed_dir = Config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = processed_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"Saved vocab to {vocab_path}")

    # 3. 加载 labels
    labels = load_labels(Config.LABEL_CSV)

    # 4. 建 dataloader
    loader = create_dataloader(log_dir, vocab, labels=labels, batch_size=4, shuffle=True)

    # 5. 打印一个 batch 看看
    for batch in loader:
        input_ids, y, fnames = batch
        print("Batch input_ids shape:", input_ids.shape)  # (batch, seq_len)
        print("Batch labels:", y)
        print("Filenames:", fnames)
        break

if __name__ == "__main__":
    main()
