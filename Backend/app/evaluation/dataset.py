from __future__ import annotations

import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET


DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class GroundTruthRow:
    code: str
    antibiotic: str
    diameter_mm: Optional[float]
    interpretation: Optional[str]
    resistant_breakpoint_mm: Optional[str]
    susceptible_breakpoint_mm: Optional[str]


@dataclass
class DatasetImageRecord:
    image_id: str
    source_group: str
    provenance: str
    analyzable: bool
    image_path: str
    measured_reference_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    ground_truth_rows: List[GroundTruthRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ground_truth_rows"] = [asdict(row) for row in self.ground_truth_rows]
        return payload


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _parse_table_cell_text(cell) -> str:
    texts = []
    for paragraph in cell.findall(".//w:p", DOCX_NS):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", DOCX_NS)]
        value = "".join(parts).strip()
        if value:
            texts.append(value)
    return " ".join(texts).strip()


def parse_dryad_table(docx_path: Path) -> List[GroundTruthRow]:
    with zipfile.ZipFile(docx_path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))

    table = document.find(".//w:tbl", DOCX_NS)
    if table is None:
        return []

    rows: List[GroundTruthRow] = []
    for row_index, table_row in enumerate(table.findall("./w:tr", DOCX_NS)):
        cells = [_parse_table_cell_text(cell) for cell in table_row.findall("./w:tc", DOCX_NS)]
        if row_index == 0 or len(cells) < 4:
            continue

        diameter_value: Optional[float] = None
        try:
            diameter_value = float(cells[2])
        except (TypeError, ValueError):
            diameter_value = None

        rows.append(
            GroundTruthRow(
                code=cells[0].strip().upper(),
                antibiotic=cells[1].strip(),
                diameter_mm=diameter_value,
                interpretation=cells[3].strip().upper() or None,
                resistant_breakpoint_mm=cells[4].strip() if len(cells) > 4 else None,
                susceptible_breakpoint_mm=cells[5].strip() if len(cells) > 5 else None,
            )
        )
    return rows


def _sample_id_from_name(name: str, suffix: str) -> str:
    normalized = name
    if normalized.lower().endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized.strip()


def _build_dryad_records(test_root: Path) -> List[DatasetImageRecord]:
    dryad_root = test_root / "dryad_sirscan"
    originals_root = dryad_root / "images_original"
    measured_root = dryad_root / "images_measured"
    tables_root = dryad_root / "Tables"
    records: List[DatasetImageRecord] = []

    for image_path in sorted(_iter_image_files(originals_root)):
        sample_id = _sample_id_from_name(image_path.name, ". original.jpg")
        measured_path = measured_root / f"{sample_id}. measured.jpg"
        table_path = tables_root / f"Table {sample_id}..docx"
        ground_truth_rows = parse_dryad_table(table_path) if table_path.exists() else []
        notes = []
        if not measured_path.exists():
            notes.append("Measured reference image is missing.")
        if not table_path.exists():
            notes.append("Ground-truth table is missing.")
        records.append(
            DatasetImageRecord(
                image_id=sample_id,
                source_group="dryad_sirscan",
                provenance="dryad_original",
                analyzable=True,
                image_path=str(image_path),
                measured_reference_path=str(measured_path) if measured_path.exists() else None,
                ground_truth_path=str(table_path) if table_path.exists() else None,
                ground_truth_rows=ground_truth_rows,
                notes=notes,
            )
        )

    for image_path in sorted(_iter_image_files(measured_root)):
        sample_id = _sample_id_from_name(image_path.name, ". measured.jpg")
        records.append(
            DatasetImageRecord(
                image_id=f"{sample_id}_measured",
                source_group="dryad_sirscan",
                provenance="dryad_measured_reference",
                analyzable=False,
                image_path=str(image_path),
                notes=["Reference-only measured overlay; excluded from inference runs."],
            )
        )

    return records


def _build_custom_records(test_root: Path) -> List[DatasetImageRecord]:
    custom_root = test_root / "custom_plates"
    records: List[DatasetImageRecord] = []
    for image_path in sorted(_iter_image_files(custom_root)):
        image_id = re.sub(r"[^A-Za-z0-9]+", "_", image_path.stem).strip("_") or image_path.stem
        records.append(
            DatasetImageRecord(
                image_id=image_id,
                source_group="custom_plates",
                provenance="custom_plate",
                analyzable=True,
                image_path=str(image_path),
            )
        )
    return records


def index_test_images(test_root: Path) -> List[DatasetImageRecord]:
    records = []
    records.extend(_build_custom_records(test_root))
    records.extend(_build_dryad_records(test_root))
    return records

