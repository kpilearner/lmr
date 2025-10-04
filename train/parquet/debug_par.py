import argparse
import os
from typing import Any

from datasets import load_dataset


def save_image(img: Any, path: str) -> None:
    """Persist a PIL image or numpy array to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(img, "save"):
        img.save(path)
    else:
        from PIL import Image
        Image.fromarray(img).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parquet samples")
    parser.add_argument("--parquet", required=True, help="Path to parquet file or glob pattern")
    parser.add_argument("--output", default="debug_samples", help="Directory to dump sample images")
    parser.add_argument("--num", type=int, default=2, help="Number of rows to export")
    parser.add_argument("--cache", default=None, help="Optional datasets cache directory")
    args = parser.parse_args()

    load_kwargs = {"data_files": os.path.abspath(args.parquet)}
    if args.cache:
        os.environ["HF_DATASETS_CACHE"] = args.cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache
        load_kwargs["cache_dir"] = args.cache
        print(f"Datasets cache redirected to: {args.cache}")

    dataset = load_dataset("parquet", **load_kwargs)["train"]
    print("Columns:", dataset.column_names)
    sample_count = min(args.num, len(dataset))
    print(f"Inspecting {sample_count} / {len(dataset)} samples")

    for idx in range(sample_count):
        sample = dataset[idx]
        print(f"\nSample {idx}:")
        for key, value in sample.items():
            if hasattr(value, "size"):
                print(f"  {key}: image size {value.size}")
            else:
                print(f"  {key}: {type(value)}")

        for image_key in ["src_img", "edited_img", "panoptic_img"]:
            if image_key in sample:
                filename = os.path.join(args.output, f"sample{idx}_{image_key}.png")
                save_image(sample[image_key], filename)
                print(f"  -> saved {image_key} to {filename}")


if __name__ == "__main__":
    main()
