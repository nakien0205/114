"""
Audit module for fire and smoke datasets: scanning, validation, statistics, and reporting.
"""

from src.fire_audit.audit.validator import (
    validate_bbox_line,
    validate_label_file,
    validate_image_file,
)
from src.fire_audit.audit.scanner import (
    DatasetScanner,
    scan_dataset,
)
from src.fire_audit.audit.stats import (
    StatsCalculator,
    calculate_dataset_stats,
    aggregate_stats,
)
from src.fire_audit.audit.reporter import (
    ReportGenerator,
    generate_markdown_report,
    generate_json_report,
    save_reports,
)

__all__ = [
    "validate_bbox_line",
    "validate_label_file",
    "validate_image_file",
    "DatasetScanner",
    "scan_dataset",
    "StatsCalculator",
    "calculate_dataset_stats",
    "aggregate_stats",
    "ReportGenerator",
    "generate_markdown_report",
    "generate_json_report",
    "save_reports",
]
