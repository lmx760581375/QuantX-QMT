"""FastAPI app for the local QuantX workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from quantx.server.services import (
    BacktestTaskService,
    ConfigService,
    DailyRunService,
    DataService,
    MetaService,
    PatternAnalysisService,
    ReportService,
)


app = FastAPI(title="QuantX Workspace", version="0.1.0")
configs = ConfigService()
reports = ReportService()
daily_runs = DailyRunService()
tasks = BacktestTaskService()
data_service = DataService()
meta_service = MetaService()
patterns = PatternAnalysisService()


class ConfigWriteRequest(BaseModel):
    content: str


class ConfigValidateRequest(BaseModel):
    content: str


class BacktestRequest(BaseModel):
    config_path: str
    symbol_limit: Optional[int] = None


class DataStatusRequest(BaseModel):
    provider_uri: str = "data/qlib_data_fixed"


class PatternBuildRequest(BaseModel):
    runs: list[str]
    analysis_id: Optional[str] = None
    pre_n: int = 40
    post_n: int = 15
    n_clusters: int = 8
    models: Optional[list[str]] = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = Path(__file__).resolve().parent / "static" / "workspace.html"
    return html.read_text(encoding="utf-8")


@app.get("/api/configs")
def list_configs():
    return configs.list_configs()


@app.get("/api/configs/{config_path:path}")
def read_config(config_path: str):
    try:
        return configs.read_config(config_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/configs/{config_path:path}")
def write_config(config_path: str, request: ConfigWriteRequest):
    try:
        return configs.write_config(config_path, request.content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/configs/{config_path:path}")
def delete_config(config_path: str):
    try:
        return configs.delete_config(config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/configs/validate")
def validate_config(request: ConfigValidateRequest):
    return configs.validate_config(request.content)


@app.post("/api/backtests")
def submit_backtest(request: BacktestRequest):
    try:
        return tasks.submit(request.config_path, symbol_limit=request.symbol_limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/backtests/{task_id}")
def get_backtest(task_id: str):
    try:
        return tasks.get(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=task_id) from exc


@app.get("/api/backtests/{task_id}/logs")
def get_backtest_logs(task_id: str):
    try:
        return {"task_id": task_id, "logs": tasks.get(task_id)["logs"]}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=task_id) from exc


@app.get("/api/reports")
def list_reports():
    return reports.list_reports()


@app.get("/api/reports/{run_id}")
def read_report(run_id: str):
    try:
        return reports.read_report(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports/{run_id}/{artifact}")
def read_report_artifact(run_id: str, artifact: str):
    try:
        return reports.read_artifact(run_id, artifact)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports/{run_id}/symbols/{symbol}")
def read_report_symbol(run_id: str, symbol: str):
    try:
        return reports.read_symbol_detail(run_id, symbol)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/daily-runs")
def list_daily_runs():
    return daily_runs.list_runs()


@app.get("/api/daily-runs/{date_key}/{profile}")
def read_daily_run(date_key: str, profile: str):
    try:
        return daily_runs.read_run(date_key, profile)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/data/status")
def data_status(provider_uri: str = "data/qlib_data_fixed"):
    return data_service.status(provider_uri)


@app.get("/api/pattern-analyses")
def list_pattern_analyses():
    return patterns.list_analyses()


@app.post("/api/pattern-analyses")
def build_pattern_analysis(request: PatternBuildRequest):
    try:
        return patterns.build(
            runs=request.runs,
            analysis_id=request.analysis_id,
            pre_n=request.pre_n,
            post_n=request.post_n,
            n_clusters=request.n_clusters,
            models=request.models,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pattern-analyses/{analysis_id}")
def read_pattern_analysis(analysis_id: str):
    try:
        return patterns.read_analysis(analysis_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pattern-analyses/{analysis_id}/clusters/{cluster_id}/samples")
def read_pattern_cluster_samples(analysis_id: str, cluster_id: str, limit: int = 100):
    try:
        return patterns.cluster_samples(analysis_id, cluster_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pattern-analyses/{analysis_id}/similar")
def read_similar_pattern_samples(
    analysis_id: str,
    sample_id: str,
    top_k: int = 20,
    same_cluster_only: bool = False,
):
    try:
        return patterns.similar(analysis_id, sample_id, top_k=top_k, same_cluster_only=same_cluster_only)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pattern-analyses/{analysis_id}/figures/{figure_path:path}")
def read_pattern_figure(analysis_id: str, figure_path: str):
    try:
        return FileResponse(patterns.figure_path(analysis_id, figure_path))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/data/health-check")
def data_health_check(request: DataStatusRequest):
    return data_service.health_check(request.provider_uri)


@app.post("/api/data/update-plan")
def data_update_plan(request: DataStatusRequest):
    return data_service.update_plan(request.provider_uri)


@app.get("/api/meta/symbols/{symbol}")
def meta_symbol(symbol: str):
    return meta_service.symbol(symbol)


@app.get("/api/meta/industries")
def meta_industries():
    return meta_service.industries()


@app.get("/api/meta/concepts")
def meta_concepts():
    return meta_service.concepts()


def main() -> None:
    import uvicorn

    uvicorn.run("quantx.server.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
