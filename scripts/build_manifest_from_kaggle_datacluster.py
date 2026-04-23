from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ALPR manifest.csv from dataclusterlabs/indian-number-plates-dataset"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Root directory where Kaggle dataset is extracted",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/manifest.csv",
        help="Output manifest path",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include objects that do not have number_plate_text by using UNKNOWN labels",
    )
    return parser.parse_args()


def _safe_int(value: str | None) -> int:
    if value is None:
        return 0
    return int(round(float(value)))


def _normalize_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    return cleaned


def _extract_plate_text(obj: ET.Element) -> str | None:
    attributes = obj.find("attributes")
    if attributes is None:
        return None

    for attr in attributes.findall("attribute"):
        name = (attr.findtext("name") or "").strip().lower()
        value = (attr.findtext("value") or "").strip()
        if name == "number_plate_text" and value:
            return _normalize_plate_text(value)
    return None


def _find_image_path(raw_dir: Path, filename: str) -> Path | None:
    candidates = [
        raw_dir / "number_plate_images_ocr" / "number_plate_images_ocr" / filename,
        raw_dir / "Indian_Number_Plates" / "Sample_Images" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    matches = list(raw_dir.rglob(filename))
    if matches:
        return matches[0].resolve()
    return None


def parse_xml_to_rows(xml_path: Path, raw_dir: Path, include_unlabeled: bool) -> list[dict[str, object]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = (root.findtext("filename") or "").strip()
    if not filename:
        return []

    image_path = _find_image_path(raw_dir, filename)
    if image_path is None:
        return []

    rows: list[dict[str, object]] = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() != "number_plate":
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        x_min = _safe_int(bnd.findtext("xmin"))
        y_min = _safe_int(bnd.findtext("ymin"))
        x_max = _safe_int(bnd.findtext("xmax"))
        y_max = _safe_int(bnd.findtext("ymax"))

        plate_text = _extract_plate_text(obj)
        if not plate_text:
            if not include_unlabeled:
                continue
            plate_text = "UNKNOWN"

        if x_max <= x_min or y_max <= y_min:
            continue

        rows.append(
            {
                "image_path": str(image_path),
                "plate_text": plate_text,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)

    xml_dirs = [
        raw_dir / "number_plate_annos_ocr" / "number_plate_annos_ocr",
        raw_dir / "Annotations" / "Annotations",
    ]

    xml_files: list[Path] = []
    for d in xml_dirs:
        if d.exists():
            xml_files.extend(sorted(d.glob("*.xml")))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found under expected folders inside: {raw_dir.resolve()}"
        )

    all_rows: list[dict[str, object]] = []
    for xml_file in xml_files:
        all_rows.extend(parse_xml_to_rows(xml_file, raw_dir, include_unlabeled=args.include_unlabeled))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "plate_text", "x_min", "y_min", "x_max", "y_max"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[manifest] xml_files={len(xml_files)}")
    print(f"[manifest] rows={len(all_rows)}")
    print(f"[manifest] output={output_path.resolve()}")


if __name__ == "__main__":
    main()
