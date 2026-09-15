"""Import user-supplied results with provenance, without network access."""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from sports_ai_bot.collect.historical import LEAGUES
from sports_ai_bot.storage import Store
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.team_names import canonical_team_name


def import_history(path: Path, league: str, source_url: str, permission_note: str) -> int:
    if league not in LEAGUES.values():
        raise ValueError("Liga desconocida; usa coverage para consultar identificadores.")
    url = urlsplit(source_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("La procedencia debe ser una URL HTTPS sin credenciales.")
    if not permission_note.strip():
        raise ValueError("Indica el permiso de uso de los datos; no implica verificacion legal.")
    payload = Path(path).read_bytes()
    frame = pd.read_csv(io.BytesIO(payload), keep_default_na=False)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if frame.empty or not set(required).issubset(frame):
        raise ValueError(f"CSV vacio o incompleto; columnas: {', '.join(required)}")
    dates = []
    now = datetime.now(timezone.utc)
    for value in frame["Date"]:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Date requiere ISO 8601 con zona horaria.") from exc
        if stamp.tzinfo is None or stamp.astimezone(timezone.utc) >= now:
            raise ValueError("Los resultados requieren fecha pasada con zona horaria.")
        dates.append(stamp.astimezone(timezone.utc).isoformat())
    frame["Date"] = dates
    for column in ("HomeTeam", "AwayTeam"):
        frame[column] = frame[column].map(lambda x: " ".join(str(x).split()))
        if frame[column].eq("").any():
            raise ValueError("Los nombres de equipos no pueden estar vacios.")
        frame[column] = frame[column].map(lambda x: canonical_team_name(league, x) or x)
    if frame["HomeTeam"].str.casefold().eq(frame["AwayTeam"].str.casefold()).any():
        raise ValueError("Local y visitante deben ser distintos.")
    for column in ("FTHG", "FTAG", "HC", "AC"):
        if column not in frame:
            continue
        empty = frame[column].astype(str).str.strip().eq("")
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)
        if column in ("HC", "AC"):
            valid |= empty
        if not valid.all():
            raise ValueError(f"{column} debe contener enteros no negativos; corners puede estar vacio.")
        frame[column] = values
    columns = required + [c for c in ("HC", "AC") if c in frame]
    frame = frame[columns].drop_duplicates()
    if frame.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Resultados contradictorios para el mismo partido.")
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    destination = settings.raw_dir / f"{league}_{digest}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    manifest_dir = settings.data_dir / "imports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{league}_{digest}.json"
    temp_manifest = manifest.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps({
        "sha256": digest, "league": league, "source_url": source_url,
        "permission_note": permission_note.strip(), "imported_at": now.isoformat(),
        "rows": len(frame), "file": destination.name, "method": "local_import",
    }, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest)
    store = Store(settings.data_dir)
    store.set_status("picks", [])
    store.set_status("observation_picks", [])
    store.set_status("generation", {"code": "cache_invalid", "observation": False})
    return len(frame)
