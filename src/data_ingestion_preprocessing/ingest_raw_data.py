"""Ingest and clean annual MFCU statistical workbooks into CSV outputs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

NS_MAIN = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
WB_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

CANONICAL_HEADER_MAP = {
    "state": "jurisdiction",
    "total_investigations": "total_investigations",
    "fraud_investigations": "fraud_investigations",
    "abuse_neglect_investigations": "abuse_neglect_investigations",
    "total_indictments": "total_indictments",
    "fraud_indictments": "fraud_indictments",
    "abuse_neglect_indictments": "abuse_neglect_indictments",
    "total_convictions": "total_convictions",
    "fraud_convictions": "fraud_convictions",
    "abuse_neglect_convictions": "abuse_neglect_convictions",
    "civil_settlements_and_judgments": "civil_settlements_and_judgments",
    "total_recoveries": "total_recoveries",
    "total_criminal_recoveries": "total_criminal_recoveries",
    "total_civil_recoveries": "total_civil_recoveries",
    "civil_recoveries_global": "civil_recoveries_global",
    "civil_recoveries_other": "civil_recoveries_other",
    "mfcu_grant_expenditures": "mfcu_grant_expenditures",
    "total_medicaid_expenditures": "total_medicaid_expenditures",
    "staff_on_board": "staff_on_board",
}

TERRITORIES = {"Puerto Rico", "U.S. Virgin Islands"}
DISTRICTS = {"District of Columbia"}

JURISDICTION_OUTPUT_COLUMNS = [
    "fiscal_year",
    "jurisdiction",
    "jurisdiction_type",
    "total_investigations",
    "fraud_investigations",
    "abuse_neglect_investigations",
    "total_indictments",
    "fraud_indictments",
    "abuse_neglect_indictments",
    "total_convictions",
    "fraud_convictions",
    "abuse_neglect_convictions",
    "civil_settlements_and_judgments",
    "total_recoveries",
    "total_criminal_recoveries",
    "total_civil_recoveries",
    "civil_recoveries_global",
    "civil_recoveries_other",
    "mfcu_grant_expenditures",
    "total_medicaid_expenditures",
    "staff_on_board",
]

NATIONAL_OUTPUT_COLUMNS = [
    "fiscal_year",
    "jurisdiction",
    "total_investigations",
    "fraud_investigations",
    "abuse_neglect_investigations",
    "total_indictments",
    "fraud_indictments",
    "abuse_neglect_indictments",
    "total_convictions",
    "fraud_convictions",
    "abuse_neglect_convictions",
    "civil_settlements_and_judgments",
    "total_recoveries",
    "total_criminal_recoveries",
    "total_civil_recoveries",
    "civil_recoveries_global",
    "civil_recoveries_other",
    "mfcu_grant_expenditures",
    "total_medicaid_expenditures",
    "staff_on_board",
]


def col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - 64)
    return value - 1


def get_shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    values = []
    for shared_string in root.findall("a:si", NS_MAIN):
        text = "".join(
            text_node.text or ""
            for text_node in shared_string.iterfind(".//a:t", NS_MAIN)
        )
        values.append(text)
    return values


def get_first_sheet_path(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    relationships = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall("r:Relationship", NS_REL)
    }
    first_sheet = workbook.find("a:sheets/a:sheet", NS_MAIN)
    if first_sheet is None:
        raise ValueError("Workbook does not contain any sheets.")

    rel_id = first_sheet.attrib.get(f"{{{WB_REL_NS}}}id")
    if not rel_id:
        raise ValueError("First worksheet is missing a relationship id.")

    target = rel_map[rel_id]
    return target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            text_node.text or ""
            for text_node in cell.iterfind(".//a:t", NS_MAIN)
        ).strip()

    value_node = cell.find("a:v", NS_MAIN)
    if value_node is None:
        return "".join(
            text_node.text or ""
            for text_node in cell.iterfind(".//a:t", NS_MAIN)
        ).strip()

    raw_value = (value_node.text or "").strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)].strip()
        except (IndexError, ValueError):
            return raw_value
    return raw_value


def read_nonempty_rows(workbook_path: Path) -> list[list[str]]:
    with ZipFile(workbook_path) as zf:
        shared_strings = get_shared_strings(zf)
        sheet_path = get_first_sheet_path(zf)
        worksheet = ET.fromstring(zf.read(sheet_path))

        rows = []
        for row in worksheet.findall(".//a:sheetData/a:row", NS_MAIN):
            row_values: dict[int, str] = {}
            for cell in row.findall("a:c", NS_MAIN):
                reference = cell.attrib.get("r", "")
                index = col_index(reference) if reference else len(row_values)
                row_values[index] = cell_value(cell, shared_strings)

            if any(value for value in row_values.values()):
                max_index = max(row_values) if row_values else -1
                rows.append([row_values.get(i, "") for i in range(max_index + 1)])
        return rows


def standardize_header(header: str) -> str:
    cleaned = re.sub(r"\d+", "", header.strip())
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("/", " ")
    cleaned = cleaned.replace(",", " ")
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return CANONICAL_HEADER_MAP.get(cleaned, cleaned)


def extract_fiscal_year(path: Path) -> int:
    match = re.search(r"FY_(\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not extract fiscal year from {path.name}")
    return int(match.group(1))


def normalize_jurisdiction(name: str) -> str:
    collapsed = " ".join(name.strip().split())
    upper = collapsed.upper()
    normalized_map = {
        "DISTRICT OF COLUMBIA": "District of Columbia",
        "DISTRICT OF COLUMBIA.": "District of Columbia",
        "PUERTO RICO": "Puerto Rico",
        "U.S. VIRGIN ISLANDS": "U.S. Virgin Islands",
        "US VIRGIN ISLANDS": "U.S. Virgin Islands",
        "TOTAL :": "Total",
        "TOTAL": "Total",
        "GRAND TOTAL": "Grand Total",
    }
    if upper in normalized_map:
        return normalized_map[upper]
    return collapsed.title()


def classify_row(label: str) -> str:
    lower = label.lower()
    if any(
        marker in lower
        for marker in (
            "recoveries are defined",
            "information in this chart",
            "staff on board is defined",
            "all information is current",
            "global recoveries derive",
        )
    ):
        return "footnote"
    if label and label[0].isdigit():
        return "footnote"
    if lower in {"total", "total :", "grand total"}:
        return "national_total"
    return "jurisdiction"


def convert_value(value: str) -> str:
    text = value.strip()
    if text == "":
        return ""
    normalized = text.replace(",", "").replace("$", "")
    try:
        number = float(normalized)
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def jurisdiction_type(name: str) -> str:
    if name in TERRITORIES:
        return "territory"
    if name in DISTRICTS:
        return "district"
    return "state"


def build_record(headers: list[str], row: list[str], fiscal_year: int) -> dict[str, str]:
    padded = row + [""] * (len(headers) - len(row))
    record = dict(zip(headers, padded))
    standardized = {standardize_header(key): value for key, value in record.items()}
    standardized["fiscal_year"] = str(fiscal_year)
    standardized["jurisdiction"] = normalize_jurisdiction(
        standardized.get("jurisdiction", "")
    )

    for field in list(standardized):
        if field not in {"jurisdiction", "fiscal_year"}:
            standardized[field] = convert_value(standardized[field])
    return standardized


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    workbook_paths = sorted(RAW_DIR.glob("FY_*_MFCU_Statistical_Chart.xlsx"))
    if not workbook_paths:
        raise FileNotFoundError(f"No raw workbooks found in {RAW_DIR}")

    jurisdiction_rows: list[dict[str, str]] = []
    national_rows: list[dict[str, str]] = []
    profile_rows: list[dict[str, str]] = []

    for workbook_path in workbook_paths:
        fiscal_year = extract_fiscal_year(workbook_path)
        rows = read_nonempty_rows(workbook_path)
        if len(rows) < 2:
            raise ValueError(f"Workbook {workbook_path.name} does not contain a header row.")

        raw_headers = rows[1]
        data_row_count = 0
        found_total = False

        for row in rows[2:]:
            if not row or not row[0].strip():
                continue

            row_kind = classify_row(row[0].strip())
            if row_kind == "footnote":
                continue

            record = build_record(raw_headers, row, fiscal_year)
            if row_kind == "national_total":
                national_rows.append(record)
                found_total = True
                continue

            record["jurisdiction_type"] = jurisdiction_type(record["jurisdiction"])
            jurisdiction_rows.append(record)
            data_row_count += 1

        profile_rows.append(
            {
                "fiscal_year": str(fiscal_year),
                "source_file": workbook_path.name,
                "raw_header_count": str(len(raw_headers)),
                "jurisdiction_row_count": str(data_row_count),
                "national_total_found": "yes" if found_total else "no",
            }
        )

    jurisdiction_rows.sort(key=lambda row: (int(row["fiscal_year"]), row["jurisdiction"]))
    national_rows.sort(key=lambda row: int(row["fiscal_year"]))
    profile_rows.sort(key=lambda row: int(row["fiscal_year"]))

    write_csv(
        PROCESSED_DIR / "medicaid_jurisdiction_level.csv",
        jurisdiction_rows,
        JURISDICTION_OUTPUT_COLUMNS,
    )
    write_csv(
        PROCESSED_DIR / "medicaid_national_totals.csv",
        national_rows,
        NATIONAL_OUTPUT_COLUMNS,
    )
    write_csv(
        PROCESSED_DIR / "ingestion_profile.csv",
        profile_rows,
        [
            "fiscal_year",
            "source_file",
            "raw_header_count",
            "jurisdiction_row_count",
            "national_total_found",
        ],
    )

    print("Created cleaned datasets:")
    print("-", PROCESSED_DIR / "medicaid_jurisdiction_level.csv")
    print("-", PROCESSED_DIR / "medicaid_national_totals.csv")
    print("-", PROCESSED_DIR / "ingestion_profile.csv")


if __name__ == "__main__":
    main()
