"""
Build a single-turn QA dataset from Imagenette for toy VLM training.

Output: JSON list of {"image": rel_path, "question": "<image>\\n...", "answer": "..."}
One image produces multiple QA samples (identification, yes/no variants, category).
"""

import json
import os
import random
from pathlib import Path

IMAGENETTE_ROOT = "/home/pensativo/datasets/imagenette/imagenette2"
OUT_DIR = "/home/pensativo/code/vanilla-vit-gpt/data"

CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer spaniel",
    "n02979186": "cassette player",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

ANIMALS = {"tench", "English springer spaniel"}
VEHICLES = {"garbage truck"}
INSTRUMENTS = {"French horn"}

IDENTIFY_Q = [
    "What is in this image?",
    "What do you see in the picture?",
    "What object is shown here?",
    "What is depicted in the image?",
    "Describe the main subject of this image.",
]

YESNO_POS_Q = [
    "Is this a {c}?",
    "Does this image show a {c}?",
    "Is there a {c} in the picture?",
]

YESNO_NEG_Q = [
    "Is this a {other}?",
    "Does this image show a {other}?",
    "Is there a {other} in the picture?",
]

ANIMAL_Q = [
    "Is this an animal?",
    "Is there a living creature in the image?",
    "Is the subject of this image alive?",
]

VEHICLE_Q = [
    "Is this a vehicle?",
    "Is there a vehicle in the image?",
]


def article(name):
    return "An" if name[0].lower() in "aeiou" else "A"


def a_lower(name):
    return article(name).lower()


def gen_qa_for_image(class_name, rng):
    pairs = []
    others = [c for c in CLASSES.values() if c != class_name]
    other = rng.choice(others)

    # 1. identification
    q = rng.choice(IDENTIFY_Q)
    a = f"{article(class_name)} {class_name}."
    pairs.append((q, a))

    # 2. positive yes/no
    q = rng.choice(YESNO_POS_Q).format(c=class_name)
    a = f"Yes, this is {a_lower(class_name)} {class_name}."
    pairs.append((q, a))

    # 3. negative yes/no
    q = rng.choice(YESNO_NEG_Q).format(other=other)
    a = f"No, this is {a_lower(class_name)} {class_name}, not {a_lower(other)} {other}."
    pairs.append((q, a))

    # 4. animal/not
    q = rng.choice(ANIMAL_Q)
    a = "Yes, it is an animal." if class_name in ANIMALS else "No, it is not an animal."
    pairs.append((q, a))

    # 5. vehicle/not (50% of the time, to avoid over-representing yes/no)
    if rng.random() < 0.5:
        q = rng.choice(VEHICLE_Q)
        a = "Yes, it is a vehicle." if class_name in VEHICLES else "No, it is not a vehicle."
        pairs.append((q, a))

    return pairs


def build_split(split, seed):
    split_dir = Path(IMAGENETTE_ROOT) / split
    rng = random.Random(seed)
    samples = []
    per_class = {}
    for wnid, class_name in CLASSES.items():
        class_dir = split_dir / wnid
        img_paths = sorted(class_dir.glob("*.JPEG"))
        per_class[class_name] = len(img_paths)
        for img_path in img_paths:
            rel_path = str(img_path.relative_to(IMAGENETTE_ROOT))
            for q, a in gen_qa_for_image(class_name, rng):
                samples.append({
                    "image": rel_path,
                    "question": f"<image>\n{q}",
                    "answer": a,
                })
    rng.shuffle(samples)
    return samples, per_class


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, seed in (("train", 42), ("val", 43)):
        samples, per_class = build_split(split, seed)
        out_path = os.path.join(OUT_DIR, f"imagenette_qa_{split}.json")
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"[{split}] {len(samples)} QA samples from {sum(per_class.values())} images")
        print(f"         → {out_path}")
        print(f"         images per class: {per_class}")


if __name__ == "__main__":
    main()
