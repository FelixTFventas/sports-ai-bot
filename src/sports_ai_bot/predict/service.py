"""Offline picks from immutable manual quotes and verified v2 artifacts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sports_ai_bot.features import build
from sports_ai_bot.predict.pipeline import Pick, persist_picks
from sports_ai_bot.storage import BOOKMAKERS, Store
from sports_ai_bot.train.train_models import FEATURE_COLUMNS
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.team_names import canonical_team_name


MARKETS = {"Over 2.5": "target_over25", "BTTS": "target_btts"}
CACHE_KEY = "picks"
OBSERVATION_CACHE_KEY = "observation_picks"
STATUS_KEY = "generation"
BOGOTA = timezone(timedelta(hours=-5), "America/Bogota")
WARNING = (
    "Advertencia: probabilidades estimadas, no garantias. EV teorico, no rentabilidad "
    "demostrada. Cuotas manuales, no en vivo; verifica disponibilidad antes de decidir."
)
STATUS_MESSAGES = {
    "not_generated": "Todavia no se ha generado un analisis local.",
    "no_quotes": "Sin cuotas manuales frescas de eventos futuros.",
    "no_data": "Sin datos locales suficientes y recientes para ambos equipos.",
    "no_model": "Sin modelo compatible y autorizado para estos mercados y ligas.",
    "no_value": "Sin valor: ningun candidato supera los umbrales de edge y EV.",
    "ready": "Analisis local disponible.",
    "cache_invalid": "Cache caducada o invalidada; genera un nuevo analisis local.",
}
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return result.astimezone(timezone.utc)


def _historical_frame(settings) -> pd.DataFrame:
    frames = []
    for league in build.LEAGUES.values():
        for path in sorted(settings.raw_dir.glob(f"{league}_*.csv")):
            frame = build._load_raw_csv(path)
            frame["League"] = league
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return build._deduplicate_matches(pd.concat(frames, ignore_index=True)).dropna(
        subset=["FTHG", "FTAG"]
    )


def _event_features(quote: dict, history: pd.DataFrame, now: datetime, settings):
    kickoff = pd.Timestamp(_timestamp(quote["kickoff"]))
    cutoff = min(kickoff.tz_convert(None).normalize(), pd.Timestamp(now).tz_convert(None).normalize())
    matches = history[(history["League"] == quote["league"]) & (history["Date"] < cutoff)]
    states = build._build_state_from_completed_matches(matches)
    home = canonical_team_name(quote["league"], quote["home_team"]) or quote["home_team"]
    away = canonical_team_name(quote["league"], quote["away_team"]) or quote["away_team"]
    row = build._build_feature_row(quote["league"], kickoff, home, away, states)
    if row.get("data_schema_version") != build.DATA_SCHEMA_VERSION:
        return None
    if not row.get("quality_history_ready", False):
        return None
    for side in ("home", "away"):
        if any(row.get(f"quality_{side}_{kind}_matches", 0) < build.WINDOW
               for kind in ("history", "venue")):
            return None
        rest = row.get(f"{side}_rest_days")
        if rest is None or not 0 < rest <= settings.history_max_age_days:
            return None
    values = pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS).astype(float)
    required = [column for column in FEATURE_COLUMNS if "corners" not in column]
    if not np.isfinite(values[required].to_numpy()).all():
        return None
    if np.isinf(values.to_numpy()).any():
        return None
    return values


def _load_models(settings) -> dict:
    """Hash and deserialize the same bytes, avoiding a file replacement race."""
    try:
        summary = json.loads((settings.reports_dir / "training_summary.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(summary, dict):
        return {}
    models = {}
    for market, target in MARKETS.items():
        report = summary.get(target)
        if not isinstance(report, dict) or not (
            report.get("report_schema_version") == 2
            and report.get("data_schema_version") == build.DATA_SCHEMA_VERSION
            and report.get("feature_columns") == FEATURE_COLUMNS
            and report.get("artifact_available") is True
            and report.get("validation_status") in ("validated", "experimental")
        ):
            continue
        try:
            payload = (settings.models_dir / f"{target}.joblib").read_bytes()
            if hashlib.sha256(payload).hexdigest() != report.get("model_version"):
                continue
            model = joblib.load(io.BytesIO(payload))
            if list(model.feature_names_in_) != FEATURE_COLUMNS or list(model.classes_) != [0, 1]:
                continue
        except Exception:
            logger.warning("Unable to load verified artifact for %s", target, exc_info=True)
            continue
        models[market] = (model, report)
    return models


def _validated(report: dict, league: str) -> bool:
    per_league = report.get("per_league", {})
    metrics = per_league.get(league, {}) if isinstance(per_league, dict) else {}
    if not (report.get("validation_status") == "validated" and isinstance(metrics, dict)
            and metrics.get("validation_status") == "validated"):
        return False
    samples = metrics.get("samples", {})
    if not isinstance(samples, dict):
        return False
    for split in ("train", "test"):
        sample = samples.get(split, {})
        count = sample.get("n") if isinstance(sample, dict) else None
        if type(count) is not int or count <= 0:
            return False
    return True


def generate_picks(observation: bool = False, limit: int = 5) -> list[Pick]:
    """Generate locally; observation snapshots never populate the normal cache."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    settings = get_settings()
    store = Store(settings.data_dir)
    now = _now()
    quotes = store.quotes(max_age_minutes=settings.quote_max_age_minutes)
    models = _load_models(settings) if quotes else {}
    eligible = [q for q in quotes if q["market"] in models and (
        observation or _validated(models[q["market"]][1], q["league"]))]
    history = pd.DataFrame()
    if eligible:
        try:
            history = _historical_frame(settings)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Unable to read local history", exc_info=True)
    features = {}
    probabilities = {}
    best = {}
    data_ready = evaluated = 0
    for quote in eligible:
        event = quote["event_id"]
        if event not in features:
            features[event] = (
                _event_features(quote, history, now, settings) if not history.empty else None
            )
        values = features[event]
        if values is None:
            continue
        data_ready += 1
        model, report = models[quote["market"]]
        key = (event, quote["market"])
        if key not in probabilities:
            try:
                probability = float(model.predict_proba(values)[0, 1])
                probabilities[key] = probability if np.isfinite(probability) and 0 <= probability <= 1 else None
            except Exception:
                logger.warning("Inference failed for %s", key, exc_info=True)
                probabilities[key] = None
        probability = probabilities[key]
        if probability is None:
            continue
        evaluated += 1
        odd = float(quote["odd"])
        edge, ev = probability - 1 / odd, probability * odd - 1
        if edge < settings.picks_min_edge or ev < settings.picks_min_ev:
            continue
        pick = Pick(
            match_date=_timestamp(quote["kickoff"]).astimezone(BOGOTA).date().isoformat(),
            home_team=quote["home_team"], away_team=quote["away_team"],
            match_label=f"{quote['home_team']} vs {quote['away_team']}",
            league=quote["league"], market=quote["market"], selection=quote["selection"],
            line=quote["line"], probability=probability, confidence="Estimada",
            model_name=f"{report['selected_model']}@{report['model_version']}",
            factors=["Historial local de ambos equipos; features-v2."],
            odd=odd, implied_probability=1 / odd, edge=edge, expected_value=ev,
            bookmaker=quote["bookmaker"], is_experimental=observation,
            event_id=event, quote_id=quote["quote_id"], kickoff=quote["kickoff"],
            quoted_at=quote["quoted_at"], source_url=quote["source_url"], source=quote["source"],
            model_version=report["model_version"], generated_at=now.isoformat(),
        )
        if event not in best or ev > best[event].expected_value:
            best[event] = pick
    picks = sorted(best.values(), key=lambda p: (-p.expected_value, p.event_id))[:limit]
    if picks:
        store.record_predictions(picks)
        persist_picks(picks)
    code = ("ready" if picks else "no_quotes" if not quotes else "no_model" if not eligible
            else "no_data" if not data_ready else "no_model" if not evaluated else "no_value")
    store.set_status(OBSERVATION_CACHE_KEY if observation else CACHE_KEY, [asdict(p) for p in picks])
    store.set_status(STATUS_KEY, {
        "code": code, "observation": observation, "generated_at": now.isoformat(),
        "quotes": len(quotes), "eligible_quotes": len(eligible),
        "data_ready": data_ready, "evaluated": evaluated, "picks": len(picks),
    })
    return picks


def cached_picks() -> list[Pick]:
    """Only return normal picks still bound to a latest quote and validated artifact."""
    settings = get_settings()
    store = Store(settings.data_dir)
    cached = store.get_status(CACHE_KEY, [])
    if not cached:
        return []
    quotes = {q["quote_id"]: q for q in store.quotes(settings.quote_max_age_minutes)}
    models = _load_models(settings)
    now = _now()
    picks = []
    for snapshot in cached if isinstance(cached, list) else []:
        try:
            pick = Pick(**snapshot)
            quote = quotes.get(pick.quote_id)
            if pick.is_experimental or quote is None or pick.market not in models:
                continue
            report = models[pick.market][1]
            if not _validated(report, pick.league) or pick.model_version != report["model_version"]:
                continue
            if not (_timestamp(pick.kickoff) > now and
                    0 <= (now - _timestamp(pick.quoted_at)).total_seconds()
                    <= settings.quote_max_age_minutes * 60):
                continue
            if any(getattr(pick, key) != quote[key] for key in (
                "event_id", "kickoff", "quoted_at", "league", "home_team", "away_team",
                "market", "selection", "line", "odd", "bookmaker", "source", "source_url",
            )):
                continue
            if pick.edge < settings.picks_min_edge or pick.expected_value < settings.picks_min_ev:
                continue
            picks.append(pick)
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    store.set_status(CACHE_KEY, [asdict(p) for p in picks])
    if not picks:
        store.set_status(STATUS_KEY, {
            "code": "cache_invalid", "observation": False, "generated_at": now.isoformat(),
            "quotes": len(quotes), "eligible_quotes": 0, "data_ready": 0,
            "evaluated": 0, "picks": 0,
        })
    return picks


def _local_time(value: str) -> str:
    return _timestamp(value).astimezone(BOGOTA).strftime("%Y-%m-%d %H:%M") + " Bogota (UTC-5)"


def format_picks(picks: list[Pick]) -> str:
    if not picks:
        return status_message()
    blocks = ["Analisis de cuotas manuales"]
    for pick in picks:
        mode = "OBSERVACION EXPERIMENTAL, no validada" if pick.is_experimental else "Modelo validado para la liga"
        blocks.append(
            f"{pick.match_label} | {pick.league}\n{_local_time(pick.kickoff)}\n"
            f"{pick.market} / {pick.selection} | {mode}\n"
            f"Probabilidad estimada: {pick.probability:.1%}\n"
            f"Probabilidad de equilibrio de la cuota: {pick.implied_probability:.1%}\n"
            f"Casa: {BOOKMAKERS.get(pick.bookmaker, pick.bookmaker)} | "
            f"Cuota observada manual: {pick.odd:.2f}\n"
            f"Observada: {_local_time(pick.quoted_at)}\n"
            f"Edge: {pick.edge:+.1%} | EV teorico: {pick.expected_value:+.1%}\n"
            f"Fuente: {pick.source_url}\nModelo: {pick.model_version}"
        )
    return "\n\n".join([*blocks, WARNING])


def format_quotes() -> str:
    settings = get_settings()
    quotes = Store(settings.data_dir).quotes(settings.quote_max_age_minutes)
    if not quotes:
        return STATUS_MESSAGES["no_quotes"]
    blocks = ["Cuotas observadas manualmente, no en vivo"]
    for quote in quotes:
        blocks.append(
            f"{quote['home_team']} vs {quote['away_team']} | {quote['league']}\n"
            f"{_local_time(quote['kickoff'])} | {quote['market']} / {quote['selection']}\n"
            f"Casa: {BOOKMAKERS.get(quote['bookmaker'], quote['bookmaker'])} | "
            f"Cuota observada manual: {quote['odd']:.2f}\n"
            f"Observada: {_local_time(quote['quoted_at'])}\nFuente: {quote['source_url']}"
        )
    return "\n\n".join([*blocks, WARNING])


def status_message() -> str:
    state = Store(get_settings().data_dir).get_status(STATUS_KEY, {})
    code = state.get("code", "not_generated")
    mode = "Observacion experimental" if state.get("observation") else "Modo validado"
    return f"{mode}: {STATUS_MESSAGES.get(code, STATUS_MESSAGES['not_generated'])}\n{WARNING}"
