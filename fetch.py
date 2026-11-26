"""Download pinned public inputs into a local cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj

BASE = "google/bert_uncased_L-2_H-128_A-2/resolve/30b0a37ccaaa32f332884b96992754e246e48c5f"
FLAN = "google/flan-t5-small/resolve/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
DATA = "datasets/deepset/prompt-injections/resolve/4f61ecb038e9c3fb77e21034b22511b523772cdd/data"

def fetch(cache):
    jobs = []
    for name, repo, files in [
        ("base", BASE, ["config.json", "vocab.txt", "model.safetensors"]),
        ("flan", FLAN, ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]),
        ("data", DATA, ["train-00000-of-00001-9564e8b05b4757ab.parquet", "test-00000-of-00001-701d16158af87368.parquet"]),
    ]:
        for file in files:
            jobs.append((cache / name / file, f"https://huggingface.co/{repo}/{file}"))

    def download(job):
        path, url = job
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".partial-" + path.suffix.lstrip("."))
        with urlopen(url, timeout=180) as source, temp.open("wb") as target:
            copyfileobj(source, target)
        temp.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    fetch(parser.parse_args().cache)
