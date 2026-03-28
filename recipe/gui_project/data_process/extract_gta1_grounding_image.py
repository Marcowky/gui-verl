import atexit
import argparse
import os
import zipfile
from bisect import bisect_right
from pathlib import Path


GROUNDING_DATASET_ROOT = Path("/home/kaiyu/Data/raw_data/HelloKKMe/grounding_dataset")
IMAGE_PART_PATHS = [
    GROUNDING_DATASET_ROOT / "image.part.aa",
    GROUNDING_DATASET_ROOT / "image.part.ab",
    GROUNDING_DATASET_ROOT / "image.part.ac",
    GROUNDING_DATASET_ROOT / "image.part.ad",
    GROUNDING_DATASET_ROOT / "image.part.ae",
    GROUNDING_DATASET_ROOT / "image.part.af",
    GROUNDING_DATASET_ROOT / "image.part.ag",
    GROUNDING_DATASET_ROOT / "image.part.ah",
    GROUNDING_DATASET_ROOT / "image.part.ai",
    GROUNDING_DATASET_ROOT / "image.part.aj",
]
ZIP_MEMBER_PREFIX = "image/"
_ARCHIVE = None


class MultiPartReader:
    def __init__(self, part_paths: list[Path]):
        self.part_paths = part_paths
        self.part_sizes = [path.stat().st_size for path in part_paths]
        self.part_offsets = [0]
        total_size = 0
        for part_size in self.part_sizes:
            total_size += part_size
            self.part_offsets.append(total_size)
        self.total_size = total_size
        self.position = 0
        self.handles = [path.open("rb") for path in part_paths]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.position + offset
        elif whence == os.SEEK_END:
            new_position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")

        if new_position < 0:
            raise ValueError(f"Negative seek is not allowed: {new_position}")

        self.position = new_position
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.total_size - self.position
        if self.position >= self.total_size or size == 0:
            return b""

        remaining = min(size, self.total_size - self.position)
        chunks = bytearray()

        while remaining > 0 and self.position < self.total_size:
            part_idx = bisect_right(self.part_offsets, self.position) - 1
            part_offset = self.position - self.part_offsets[part_idx]
            handle = self.handles[part_idx]
            handle.seek(part_offset)

            read_size = min(remaining, self.part_sizes[part_idx] - part_offset)
            chunk = handle.read(read_size)
            if not chunk:
                break

            chunks.extend(chunk)
            chunk_size = len(chunk)
            self.position += chunk_size
            remaining -= chunk_size

        return bytes(chunks)

    def close(self):
        for handle in self.handles:
            handle.close()


class GTA1GroundingArchive:
    def __init__(self, part_paths: list[Path]):
        self.reader = MultiPartReader(part_paths)
        self.zip_file = zipfile.ZipFile(self.reader)
        self.name_set = set(self.zip_file.namelist())

    def read_image_bytes(self, relative_image_path: str) -> bytes:
        member_name = build_zip_member_name(relative_image_path)
        try:
            return self.zip_file.read(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Image not found in GTA1 grounding archive: {relative_image_path}"
            ) from exc

    def read_image_size(self, relative_image_path: str) -> tuple[int, int]:
        image_bytes = self.read_image_bytes(relative_image_path)
        return get_image_size_from_bytes(image_bytes)

    def has_image(self, relative_image_path: str) -> bool:
        member_name = build_zip_member_name(relative_image_path)
        return member_name in self.name_set

    def close(self):
        self.zip_file.close()
        self.reader.close()


def build_zip_member_name(relative_image_path: str) -> str:
    relative_path = relative_image_path.strip().lstrip("/")
    if not relative_path:
        raise ValueError("relative_image_path must not be empty")
    return f"{ZIP_MEMBER_PREFIX}{relative_path}"


def get_image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            return image.size
    except ModuleNotFoundError:
        pass

    png_signature = b"\x89PNG\r\n\x1a\n"
    if image_bytes.startswith(png_signature) and len(image_bytes) >= 24:
        width = int.from_bytes(image_bytes[16:20], byteorder="big")
        height = int.from_bytes(image_bytes[20:24], byteorder="big")
        return width, height

    raise ValueError("Unsupported image format and Pillow is unavailable to decode it.")


def get_gta1_grounding_archive() -> GTA1GroundingArchive:
    global _ARCHIVE
    if _ARCHIVE is None:
        _ARCHIVE = GTA1GroundingArchive(IMAGE_PART_PATHS)
    return _ARCHIVE


def close_gta1_grounding_archive():
    global _ARCHIVE
    if _ARCHIVE is not None:
        _ARCHIVE.close()
        _ARCHIVE = None


atexit.register(close_gta1_grounding_archive)


def read_gta1_grounding_image_bytes(relative_image_path: str) -> bytes:
    archive = get_gta1_grounding_archive()
    return archive.read_image_bytes(relative_image_path)


def read_gta1_grounding_image_size(relative_image_path: str) -> tuple[int, int]:
    archive = get_gta1_grounding_archive()
    return archive.read_image_size(relative_image_path)


def has_gta1_grounding_image(relative_image_path: str) -> bool:
    archive = get_gta1_grounding_archive()
    return archive.has_image(relative_image_path)


def extract_gta1_grounding_image(relative_image_path: str, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = read_gta1_grounding_image_bytes(relative_image_path)
    output_path.write_bytes(image_bytes)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract one image from the GTA1 grounding dataset by inp.json image path."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Relative image path from inp.json, e.g. dataset/Aria-UI_Context-aware_Data/android_control/images/0-2.png",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save the extracted image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saved_path = extract_gta1_grounding_image(
        relative_image_path=args.image_path,
        output_path=args.output_path,
    )
    print(f"Saved image to {saved_path}")


if __name__ == "__main__":
    main()
