"""
Najva — a System 1, Jev-like decision API for Persian.

Mirrors the Jev API shape at:
    POST /v1/systemone
    Authorization: Bearer <API_KEY>
    {state, model, questions: {<id>: Question}}

Question types: choice / noul / score — identical to the Jev spec.
Answers: choice/score/noul with probabilities, confidence, scores.
See the API reference in STATUS.md or https://docs.typesafe.ai for the full spec.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or with multiple workers for throughput:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2

Auth (optional): set NAKVA_API_KEY env var; requests must carry
    Authorization: Bearer <Najva_API_KEY>.  Omit the env var to disable auth.
"""
import os
from typing import Any, Dict, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from laya import Router

# ---------------------------------------------------------------------- models

MODEL_ALIASES = {
    "najva-latest": "convaiinnovations/laya-multilingual",
    "najva-multilingual": "convaiinnovations/laya-multilingual",
    "najva": "convaiinnovations/laya-multilingual",
}
DEFAULT_MODEL = "najva-latest"
CHECKPOINT_REPO = "convaiinnovations/laya-multilingual"
API_VERSION = "najva-0.1.0"


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Union[str, dict, list]
    criteria: Optional[Dict[str, Any]] = None  # true/false descriptions


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Union[str, dict, list]
    criteria: Dict[str, Any] = Field(..., description="option → description")


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Union[str, dict, list]
    criteria: list  # ordered level descriptions


Question = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion]


class SystemOneRequest(BaseModel):
    state: Union[str, dict, list] = Field(..., description=
        "The content to evaluate: text string, or structured data (JSON object/array).")
    model: str = Field(..., description="Model alias: najva-latest, najva-multilingual, najva")
    questions: Dict[str, Question] = Field(..., min_items=1)


# ---------------------------------------------------------------------- app

# WHY NOT Router(preload=True):
#   preload=True downloads and builds ALL THREE laya checkpoints (~1.16B params).
#   For a Persian-only API every request is non-Latin script → the router routes
#   to multilingual 100% of the time anyway.  Mounting only the standalone
#   multilingual repo keeps one checkpoint resident (~644 MB) and makes routing
#   effectively free (script detection costs microseconds).
ROUTER = Router(models={"multilingual": CHECKPOINT_REPO}, default="multilingual")

API_KEY = os.environ.get("Najva_API_KEY")


def _auth(request: Request):
    """Optional bearer-token auth — only enforced if Najva_API_KEY is set."""
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")


app = FastAPI(title="Najva", version=API_VERSION,
              description="System 1 Jev-like decision model for Persian. "
                          "POST to /v1/systemone with {state, model, questions}.",
              docs_url="/", redoc_url="/docs")


@app.post("/v1/systemone")
def system_one(request: Request, body: SystemOneRequest):
    _auth(request)

    if body.model.lower() not in MODEL_ALIASES:
        raise HTTPException(422, f"Unknown model {body.model!r}. "
                                f"Available: {list(MODEL_ALIASES)}")

    # laya's Router.predict expects plain dicts, not Pydantic objects
    questions_dict = {k: v.dict() if hasattr(v, "dict") else dict(v) for k, v in body.questions.items()}
    raw = ROUTER.predict(body.state, questions_dict, model="multilingual")
    answers = _jevize_answers(raw["answers"])

    input_tokens = int(raw.get("usage", {}).get("input_tokens", 0))
    return {
        "model": API_VERSION,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


def _jevize_answers(raw: dict) -> dict:
    """Shape laya's answer payload to match the Jev response schema exactly.

    laya returns extra keys (confidence, action) that the Jev spec omits from
    the response — drop them so clients see exactly what the spec promises.
    """
    out = {}
    for qid, a in raw.items():
        t = a["type"]
        if t == "noul":
            out[qid] = {"type": "noul", "noul": round(float(a["noul"]), 4)}
        elif t == "choice":
            out[qid] = {
                "type": "choice",
                "choice": a["choice"],
                "probabilities": {k: round(float(v), 4) for k, v in a["probabilities"].items()},
                "confidence": round(float(a["confidence"]), 4),
            }
        elif t == "score":
            out[qid] = {
                "type": "score",
                "score": round(float(a["score"]), 4),
                "legend": a["legend"],
                "probabilities": {k: round(float(v), 4) for k, v in a["probabilities"].items()},
                "confidence": round(float(a["confidence"]), 4),
            }
        else:
            out[qid] = a
    return out


@app.get("/v1/models")
def list_models():
    """List available model aliases."""
    return {"data": [{"id": m, "object": "model", "owned_by": "najva"} for m in MODEL_ALIASES]}


@app.get("/health")
def health():
    return {"status": "ok", "model": CHECKPOINT_REPO, "loaded": list(ROUTER.loaded)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
