from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from visualcounter.config import AppConfig, load_config
from visualcounter.processing.engine import CountResult
from visualcounter.service import VisualCounterService

logger = logging.getLogger(__name__)
API_KEY_HEADER_NAME = "X-API-Key"
API_KEYS_ENV_NAME = "VISUALCOUNTER_API_KEYS"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _serialize_count(result: CountResult) -> dict[str, object]:
    return {
        "camera": result.camera,
        "roi_name": result.roi_name,
        "roi": result.roi,
        "count": result.count,
        "smoothed_count": result.smoothed_count,
        "smoothing_type": result.smoothing_type,
        "timestamp": result.timestamp,
        "timestamp_iso": datetime.fromtimestamp(result.timestamp, tz=timezone.utc).isoformat(),
        "sequence": result.sequence,
    }


def _load_api_keys() -> frozenset[str]:
    raw = os.environ.get(API_KEYS_ENV_NAME, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def create_app(config_path: str | None = None) -> FastAPI:
    resolved_path = Path(config_path or os.environ.get("VISUALCOUNTER_CONFIG", "config.yaml"))
    config: AppConfig = load_config(resolved_path)
    service = VisualCounterService(config)
    api_keys = _load_api_keys()

    if api_keys:
        logger.info("API key auth enabled with %d configured key(s)", len(api_keys))
    else:
        logger.warning(
            "API key auth is disabled because %s is not set",
            API_KEYS_ENV_NAME,
        )

    async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
        if not api_keys:
            return

        if api_key is None or not any(hmac.compare_digest(api_key, configured) for configured in api_keys):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(
        title="VisualCounter API",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(require_api_key)],
    )

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "visualcounter",
            "cameras": service.camera_names(),
            "config": str(resolved_path),
        }

    @app.get("/{camera_name}/rois")
    async def get_rois(camera_name: str) -> dict[str, object]:
        try:
            rois = service.get_rois(camera_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "camera": camera_name,
            "rois": rois,
            "default_roi": config.cameras[camera_name].default_roi,
        }

    @app.get("/{camera_name}/count")
    async def get_count(
        camera_name: str,
        roi_name: str | None = Query(default=None),
        roi: str | None = Query(default=None),
    ) -> dict[str, object]:
        try:
            result = service.get_count(camera_name, roi_name=roi_name, roi=roi)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return _serialize_count(result)

    def sse_event(data: dict[str, object], event: str = "count") -> str:
        payload = json.dumps(data, separators=(",", ":"))
        return f"event: {event}\ndata: {payload}\n\n"

    async def stream_counts(
        camera_name: str,
        roi_name: str | None,
        roi: str | None,
    ) -> AsyncIterator[str]:
        try:
            worker = service.worker(camera_name)
            service.resolve_roi(camera_name, roi_name=roi_name, roi=roi)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        last_sequence = 0
        while True:
            snapshot = await asyncio.to_thread(worker.wait_for_update, last_sequence, 15.0)
            if snapshot is None:
                error = worker.get_last_error()
                if error:
                    yield sse_event({"camera": camera_name, "error": error}, event="error")
                    continue
                yield ": heartbeat\n\n"
                continue

            try:
                result = service.get_count(camera_name, roi_name=roi_name, roi=roi)
            except (ValueError, RuntimeError):
                continue

            last_sequence = result.sequence
            yield sse_event(_serialize_count(result))

    @app.get("/{camera_name}/count/stream")
    async def stream_count(
        camera_name: str,
        roi_name: str | None = Query(default=None),
        roi: str | None = Query(default=None),
    ) -> StreamingResponse:
        return StreamingResponse(
            stream_counts(camera_name, roi_name, roi),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/{camera_name}/count/live")
    async def stream_count_live(
        camera_name: str,
        roi_name: str | None = Query(default=None),
        roi: str | None = Query(default=None),
    ) -> StreamingResponse:
        return await stream_count(camera_name=camera_name, roi_name=roi_name, roi=roi)

    return app
