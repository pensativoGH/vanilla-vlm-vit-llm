"""Prepare a small Visual Genome QA subset for VLM training.

This script:
1. downloads Visual Genome image metadata and QA metadata
2. splits by image id to avoid train/val image leakage
3. keeps 10% of QA images for training by default
4. writes Imagenette-style JSON manifests for reuse by the VQA dataloader
5. downloads only the images referenced by the sampled train/val subsets
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
import zipfile
from pathlib import Path


IMAGE_DATA_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
QUESTION_ANSWERS_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip"


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def load_zipped_json(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"Expected one JSON member in {zip_path}, found {members}")
        with zf.open(members[0]) as f:
            return json.load(f)


def build_samples(records: list[dict], image_id_to_name: dict[int, str]) -> list[dict]:
    samples: list[dict] = []
    missing_images = 0
    for item in records:
        image_id = item.get("image_id", item.get("id"))
        image_name = image_id_to_name.get(image_id)
        if image_name is None:
            missing_images += 1
            continue

        for qa in item["qas"]:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if not question or not answer:
                continue

            samples.append(
                {
                    "image": image_name,
                    "question": f"<image>\n{question}",
                    "answer": answer,
                }
            )

    if missing_images:
        print(f"Skipped {missing_images} image records without metadata matches")
    return samples


def download_images(image_records: list[dict], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    total = len(image_records)
    for i, item in enumerate(image_records, start=1):
        url = item["url"]
        file_name = Path(url).name
        out_path = image_dir / file_name
        if out_path.exists():
            continue

        if i % 500 == 1 or i == total:
            print(f"Downloading image {i}/{total}")
        urllib.request.urlretrieve(url, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/visual_genome")
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images"
    image_data_zip = data_dir / "image_data.json.zip"
    qa_zip = data_dir / "question_answers.json.zip"

    download_file(IMAGE_DATA_URL, image_data_zip)
    download_file(QUESTION_ANSWERS_URL, qa_zip)

    image_data = load_zipped_json(image_data_zip)
    qa_data = load_zipped_json(qa_zip)

    image_id_to_record = {item["image_id"]: item for item in image_data}
    image_id_to_name = {image_id: Path(item["url"]).name for image_id, item in image_id_to_record.items()}

    qa_image_records = [item for item in qa_data if item.get("image_id", item.get("id")) in image_id_to_record]
    rng = random.Random(args.seed)
    rng.shuffle(qa_image_records)

    total_images = len(qa_image_records)
    val_images = min(args.val_images, total_images)
    train_images = min(math.floor(total_images * args.train_fraction), total_images - val_images)

    val_records = qa_image_records[:val_images]
    train_records = qa_image_records[val_images:val_images + train_images]

    train_samples = build_samples(train_records, image_id_to_name)
    val_samples = build_samples(val_records, image_id_to_name)

    with open(data_dir / "visual_genome_qa_train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(data_dir / "visual_genome_qa_val.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"Train images: {len(train_records)} -> {len(train_samples)} QA samples")
    print(f"Val images: {len(val_records)} -> {len(val_samples)} QA samples")

    selected_image_records = [
        image_id_to_record[item.get("image_id", item.get("id"))]
        for item in train_records + val_records
    ]
    download_images(selected_image_records, image_dir)


if __name__ == "__main__":
    main()
