import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, TypeVar

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from solvers import (
    clear_q6_debug_info,
    get_q6_debug_info,
    solve_cot_math,
    solve_cosine_similarity,
    solve_embedding_trapdoors,
    solve_dynamic_extract,
    solve_invoice_extract,
    solve_korean_audio,
    solve_multimodal_qa,
    solve_proof_of_work,
    solve_semantic_rank,
    solve_spin_up_cli,
    solve_structured_extraction,
    solve_context_window_heist,
    solve_youtube_filter,
)



logger = logging.getLogger("ga3_router")
router = APIRouter()
T = TypeVar("T")


AIPIPE_TOKEN = None  # token handled inside solvers via env var


async def _read_json_body(request: Request) -> Dict[str, Any]:
    """Parse JSON body from a request regardless of Content-Type header."""
    try:
        raw = await request.body()
        if not raw:
            raise ValueError("Empty request body")
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


async def _run_solver(handler: Callable[[], Awaitable[T]], label: str) -> T | JSONResponse:
    start = time.time()
    try:
        result = await handler()
        elapsed = time.time() - start
        logger.info("Running %s completed in %.2fs", label, elapsed)
        return result
    except (RuntimeError, ValueError) as exc:
        elapsed = time.time() - start
        logger.warning("Running %s client error after %.2fs: %s", label, elapsed, exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception("Running %s failed after %.2fs", label, elapsed)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------
@router.get("/")
async def root():
    return {"status": "ok"}


@router.get("/docs")
async def docs_proxy():
    # FastAPI serves /docs automatically; keep this route to ensure it exists.
    return JSONResponse({"message": "OpenAPI docs at /docs"})


# ---------------------------------------------------------------------------
# Q2: Multimodal Image QA
# ---------------------------------------------------------------------------
@router.post("/q2/answer-image")
@router.post("/answer-image")
@router.post("/q2")
async def answer_image(request: Request):
    async def _handle():
        body: Dict[str, Any] = {}
        raw = await request.body()
        ctype = request.headers.get("content-type", "").lower()

        if raw:
            if "application/json" in ctype or raw[:1] in (b"{", b"["):
                try:
                    body = json.loads(raw)
                except Exception:
                    pass
            elif "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
                try:
                    form = await request.form()
                    body = dict(form)
                except Exception:
                    pass

        if not body:
            body = dict(request.query_params)

        image_b64 = body.get("image_base64") or body.get("image") or body.get("img") or ""
        question = body.get("question") or body.get("q") or ""

        if not image_b64:
            raise ValueError("'image_base64' field is required")
        if not question:
            raise ValueError("'question' field is required")

        logger.info("Running Q2 multimodal QA")
        ans = await solve_multimodal_qa(str(image_b64), str(question))
        return {"answer": str(ans)}

    return await _run_solver(_handle, "Q2")


# ---------------------------------------------------------------------------
# Q3 & Q7: Invoice Extraction (shared /extract route)
# ---------------------------------------------------------------------------
@router.post("/q3/extract")
@router.post("/extract")
@router.post("/q3")
@router.post("/q7/extract")
@router.post("/q7")
async def extract(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        if "invoice_text" in body:
            logger.info("Running Q3 fixed extract")
            return await solve_invoice_extract(body["invoice_text"])
        else:
            logger.info("Running Q7 structured extraction")
            return await solve_structured_extraction(body)

    return await _run_solver(_handle, "Q3/Q7")


# ---------------------------------------------------------------------------
# Q4: Dynamic Schema Extraction
# ---------------------------------------------------------------------------
@router.post("/q4/dynamic-extract")
@router.post("/dynamic-extract")
@router.post("/q4")
async def dynamic_extract(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        text = body.get("text", "")
        schema = body.get("schema", body.get("schema_def", {}))
        if not isinstance(schema, dict) or not schema:
            raise ValueError("'schema' must be a non-empty JSON object")
        logger.info("Running Q4 dynamic extract")
        return await solve_dynamic_extract(text, schema)

    return await _run_solver(_handle, "Q4")


# ---------------------------------------------------------------------------
# Q6: Korean Audio Dataset API
# ---------------------------------------------------------------------------
@router.post("/q6/answer-audio")
@router.post("/answer-audio")
@router.post("/q6")
async def korean_audio(request: Request):
    import base64 as _b64

    async def _handle():
        raw = await request.body()
        ctype = request.headers.get("content-type", "").lower()

        body: Dict[str, Any] = {}

        is_json = "application/json" in ctype or raw[:1] in (b"{", b"[")
        if is_json:
            try:
                body = json.loads(raw)
            except Exception:
                body = {}

        if not body.get("audio_base64"):
            audio_bytes = b""
            if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
                try:
                    form = await request.form()
                    for _k, v in form.items():
                        data = await v.read() if hasattr(v, "read") else None
                        if data and len(data) > 100:
                            audio_bytes = data
                            break
                except Exception:
                    pass

            if not audio_bytes and raw and not is_json:
                audio_bytes = raw

            if audio_bytes:
                body = {"audio_base64": _b64.b64encode(audio_bytes).decode()}

        logger.info("Running Q6 korean audio")
        return await solve_korean_audio(body)

    return await _run_solver(_handle, "Q6")


@router.get("/q6/debug")
async def q6_debug():
    return get_q6_debug_info()


@router.get("/q6/transcripts")
async def q6_transcripts():
    info = get_q6_debug_info()
    return {"transcript": info.get("transcript", ""), "source": info.get("transcript_source", "")}


@router.get("/q6/last-audio")
async def q6_last_audio():
    from fastapi.responses import Response

    info = get_q6_debug_info()
    return Response(content=info.get("raw_body", b""), media_type="application/octet-stream")


@router.get("/q6/clear-debug")
async def q6_clear_debug():
    clear_q6_debug_info()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Q8: Semantic Search Passage Ranking
# ---------------------------------------------------------------------------
@router.post("/q8/rank")
@router.post("/rank")
@router.post("/q8")
async def semantic_rank(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_semantic_rank(body)

    return await _run_solver(_handle, "Q8")


# ---------------------------------------------------------------------------
# Q9: Word-Problem Solver
# ---------------------------------------------------------------------------
@router.post("/q9/solve")
@router.post("/solve")
@router.post("/q9")
async def cot_math(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_cot_math(body)

    return await _run_solver(_handle, "Q9")


# ---------------------------------------------------------------------------
# Solver Routes (Bucket A endpoints)
# ---------------------------------------------------------------------------
@router.post("/solve/q1")
async def solve_q1(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_youtube_filter(body)

    return await _run_solver(_handle, "Q1")


@router.post("/solve/q5")
async def solve_q5(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_cosine_similarity(body)

    return await _run_solver(_handle, "Q5")


@router.post("/solve/q10")
async def solve_q10(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_proof_of_work(body)

    return await _run_solver(_handle, "Q10")


@router.post("/solve/q11")
async def solve_q11(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_context_window_heist(body)

    return await _run_solver(_handle, "Q11")


@router.post("/solve/q12")
async def solve_q12(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_spin_up_cli(body)

    return await _run_solver(_handle, "Q12")


@router.post("/solve/q13")
async def solve_q13(request: Request):
    async def _handle():
        body = await _read_json_body(request)
        return await solve_embedding_trapdoors(body)

    return await _run_solver(_handle, "Q13")


# NOTE: Any tenant/onboarding/config/status/dashboard routes are intentionally removed.

