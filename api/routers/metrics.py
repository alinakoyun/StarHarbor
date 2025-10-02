from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from api.utils.constants import MODELS_DIR
from api.services.pipeline import _lazy_boot_tabular, _TAB_MODEL, _FEATURES
from api.services.shap_utils import compute_global_importance
import json
import logging

log = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/summary")
def summary() -> Dict[str, Any]:
    mpath = Path(MODELS_DIR) / "metrics.json"
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(500, f"Failed to read metrics.json: {e}")
    return {"message": "metrics.json not found", "available": False}

@router.get("/feature-importance")
def feature_importance() -> Dict[str, Any]:
    _lazy_boot_tabular()
    if _TAB_MODEL is None or not _FEATURES:
        raise HTTPException(500, "Model not loaded.")
    gi = compute_global_importance(_TAB_MODEL, _FEATURES)
    return {"importance": gi}
