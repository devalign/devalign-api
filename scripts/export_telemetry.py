"""Export telemetry and pilot metrics to CSV for thesis documentation and analysis.

Usage:
    uv run python scripts/export_telemetry.py [output_path.csv]
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from src.shared.database import AsyncSessionLocal
from src.shared.telemetry.models import TelemetryEventModel


async def export_telemetry_to_csv(output_file: Path) -> int:
    """Fetch all telemetry events from database and write to CSV."""
    print("Connecting to database to fetch telemetry events...")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TelemetryEventModel).order_by(TelemetryEventModel.created_at.asc())
        )
        events = result.scalars().all()

    if not events:
        print("No telemetry events found in database.")
        # Create empty file with headers
        fieldnames = [
            "id",
            "created_at",
            "user_id",
            "event_type",
            "duration_ms",
            "status",
            "error_message",
            "file_size_bytes",
            "char_count",
            "word_count",
            "total_skills",
            "standard_skills",
            "custom_skills",
            "inferred_skills",
            "standardization_ratio",
            "inference_ratio",
            "affinity_score",
            "primary_cluster",
            "gaps_count",
            "seniority",
            "metadata_json",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Empty template CSV created at: {output_file}")
        return 0

    rows: list[dict[str, object]] = []
    for ev in events:
        meta = ev.metadata_ or {}
        rows.append(
            {
                "id": str(ev.id),
                "created_at": ev.created_at.isoformat() if ev.created_at else "",
                "user_id": str(ev.user_id) if ev.user_id else "",
                "event_type": ev.event_type,
                "duration_ms": ev.duration_ms,
                "status": ev.status,
                "error_message": ev.error_message or "",
                "file_size_bytes": meta.get("file_size_bytes", ""),
                "char_count": meta.get("char_count", ""),
                "word_count": meta.get("word_count", ""),
                "total_skills": meta.get("total_skills", meta.get("total_skills_phase1", "")),
                "standard_skills": meta.get(
                    "standard_skills", meta.get("standard_skills_phase1", "")
                ),
                "custom_skills": meta.get("custom_skills", meta.get("custom_skills_phase1", "")),
                "inferred_skills": meta.get("inferred_skills", ""),
                "standardization_ratio": meta.get(
                    "standardization_ratio", meta.get("standardization_ratio_phase1", "")
                ),
                "inference_ratio": meta.get("inference_ratio", ""),
                "affinity_score": meta.get("affinity_score", ""),
                "primary_cluster": meta.get("primary_cluster", ""),
                "gaps_count": meta.get("gaps_count", ""),
                "seniority": meta.get("seniority", ""),
                "metadata_json": json.dumps(meta, ensure_ascii=False),
            }
        )

    fieldnames = list(rows[0].keys())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully exported {len(rows)} telemetry events to: {output_file}")
    return len(rows)


def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target = Path(f"data/telemetry_export_{timestamp}.csv")

    asyncio.run(export_telemetry_to_csv(target))


if __name__ == "__main__":
    main()
