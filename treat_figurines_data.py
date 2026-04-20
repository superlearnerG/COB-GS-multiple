import argparse
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MASK_PATH = str((SCRIPT_DIR.parent / "data" / "figurines" / "object_mask").resolve())
DEFAULT_LABEL_MAPPING_PATH = str((SCRIPT_DIR.parent / "data" / "figurines" / "object_mask" / "label_mapping.json").resolve())
DEFAULT_DEBUG_MASK_PATH = str((SCRIPT_DIR / "tmp_obj_mask").resolve())
SOURCE_LABELS = (137, 147)
MERGED_LABEL = 142
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge figurines mask labels 137/147 into 142 and absorb zero strips between them."
    )
    parser.add_argument("--mask-path", default=DEFAULT_MASK_PATH, help="Directory containing source masks.")
    parser.add_argument(
        "--debug-mask-path",
        default=DEFAULT_DEBUG_MASK_PATH,
        help="Directory to write processed masks with original filenames.",
    )
    return parser.parse_args()


def read_mask(mask_file: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask file: {mask_file}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _crop_to_target_roi(target_mask: np.ndarray):
    ys, xs = np.where(target_mask)
    if ys.size == 0:
        return None

    height, width = target_mask.shape
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, height)
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, width)
    return y0, y1, x0, x1


def find_zero_strip_between_labels(mask: np.ndarray, label_a: int, label_b: int) -> np.ndarray:
    label_a_mask = mask == label_a
    label_b_mask = mask == label_b
    if not label_a_mask.any() or not label_b_mask.any():
        return np.zeros(mask.shape, dtype=bool)

    roi = _crop_to_target_roi(label_a_mask | label_b_mask)
    if roi is None:
        return np.zeros(mask.shape, dtype=bool)

    y0, y1, x0, x1 = roi
    max_kernel_size = min(max(y1 - y0, x1 - x0), 101)
    if max_kernel_size < 3:
        return np.zeros(mask.shape, dtype=bool)
    if max_kernel_size % 2 == 0:
        max_kernel_size -= 1

    target_mask = (label_a_mask | label_b_mask).astype(np.uint8)
    zero_mask = mask == 0
    if not zero_mask.any():
        return np.zeros(mask.shape, dtype=bool)

    adjacency_kernel = np.ones((3, 3), dtype=np.uint8)
    adjacent_to_a = cv2.dilate(label_a_mask.astype(np.uint8), adjacency_kernel, iterations=1).astype(bool)
    adjacent_to_b = cv2.dilate(label_b_mask.astype(np.uint8), adjacency_kernel, iterations=1).astype(bool)

    for kernel_size in range(3, max_kernel_size + 1, 2):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        closed_target = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
        _, closed_components = cv2.connectedComponents(closed_target, connectivity=8)
        shared_components = (set(np.unique(closed_components[label_a_mask]).tolist()) - {0}) & (
            set(np.unique(closed_components[label_b_mask]).tolist()) - {0}
        )
        if not shared_components:
            continue

        candidate_mask = closed_target.astype(bool) & (~target_mask.astype(bool)) & zero_mask
        if not candidate_mask.any():
            continue

        component_count, component_ids = cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)
        absorb_mask = np.zeros(mask.shape, dtype=bool)
        for component_id in range(1, component_count):
            component_mask = component_ids == component_id
            if np.any(component_mask & adjacent_to_a) and np.any(component_mask & adjacent_to_b):
                absorb_mask |= component_mask

        if absorb_mask.any():
            return absorb_mask

    return np.zeros(mask.shape, dtype=bool)


def process_mask(mask: np.ndarray):
    merged_mask = mask.copy()
    relabel_pixels = np.isin(mask, SOURCE_LABELS)
    merged_mask[relabel_pixels] = MERGED_LABEL

    zero_strip_mask = find_zero_strip_between_labels(mask, SOURCE_LABELS[0], SOURCE_LABELS[1])
    merged_mask[zero_strip_mask] = MERGED_LABEL

    return merged_mask, int(relabel_pixels.sum()), int(zero_strip_mask.sum())


def iter_mask_files(mask_dir: Path):
    for path in sorted(mask_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main():
    args = parse_args()
    mask_dir = Path(args.mask_path)
    debug_dir = Path(args.debug_mask_path)

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {mask_dir}")

    debug_dir.mkdir(parents=True, exist_ok=True)

    processed_files = 0
    changed_files = 0
    total_relabeled_pixels = 0
    total_absorbed_pixels = 0

    for mask_file in iter_mask_files(mask_dir):
        mask = read_mask(mask_file)
        processed_mask, relabeled_pixels, absorbed_pixels = process_mask(mask)
        output_path = debug_dir / mask_file.name
        if not cv2.imwrite(str(output_path), processed_mask):
            raise ValueError(f"Failed to write processed mask: {output_path}")

        processed_files += 1
        total_relabeled_pixels += relabeled_pixels
        total_absorbed_pixels += absorbed_pixels
        if relabeled_pixels > 0 or absorbed_pixels > 0:
            changed_files += 1
            print(
                f"{mask_file.name}: relabeled={relabeled_pixels}, absorbed_zero_strip={absorbed_pixels}"
            )

    print(
        "Finished processing "
        f"{processed_files} masks. "
        f"Changed {changed_files} files. "
        f"Relabeled pixels={total_relabeled_pixels}, absorbed zero-strip pixels={total_absorbed_pixels}."
    )


if __name__ == "__main__":
    main()
