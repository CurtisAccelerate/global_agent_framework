"""Utility to execute a pipeline against a Markdown/text query template and persist the response."""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_pipeline_declarations import (
    PROMPTS_DIR,
    USER_QUERY_DIR,
    create_prediction_pipeline,
    create_prediction_pipeline_stub,
    create_research_pipeline,
    load_user_query,
)


def _resolve_query_path(query: str) -> Path:
    candidate = Path(query)
    if candidate.is_file():
        return candidate.resolve()

    prompts_candidate = USER_QUERY_DIR / query
    if prompts_candidate.is_file():
        return prompts_candidate.resolve()

    raise FileNotFoundError(f"Could not locate query file: {query}")


def _derive_response_path(query_path: Path, pipeline_name: str, output: Optional[str]) -> Path:
    if output:
        target = Path(output)
        if not target.is_absolute():
            target = Path.cwd() / target
        return target

    responses_dir = Path.cwd() / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{query_path.stem}_{pipeline_name}_{timestamp}.md"
    return responses_dir / filename


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = cleaned.strip("-")
    return slug or fallback


def _write_stage_artifacts(stage_results: List[Any], work_dir: Path) -> None:
    if not stage_results:
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for idx, stage_result in enumerate(stage_results, start=1):
        metadata = getattr(stage_result, "metadata", None) or {}
        stage_name = metadata.get("agent_name") or f"stage_{idx}"
        slug = _slugify(stage_name, f"stage-{idx:02d}")
        file_path = work_dir / f"{idx:02d}_{slug}.md"

        success = getattr(stage_result, "success", None)
        error_text = getattr(stage_result, "error", None)
        data = getattr(stage_result, "data", None)
        tool_calls = getattr(stage_result, "tool_calls", None)

        if isinstance(data, (dict, list)):
            output_text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            output_text = str(data or "")

        lines = [
            f"# Stage {idx}: {stage_name}",
            "",
            f"- success: {success}",
        ]
        if error_text:
            lines.append(f"- error: {error_text}")
        if metadata:
            lines.append("- metadata:")
            lines.append("```")
            lines.append(json.dumps(metadata, indent=2, ensure_ascii=False))
            lines.append("```")
        if tool_calls:
            lines.append("- tool_calls:")
            lines.append("```")
            lines.append(json.dumps(tool_calls, indent=2, ensure_ascii=False))
            lines.append("```")

        lines.append("")
        lines.append("## Output")
        lines.append("```")
        lines.append(output_text)
        lines.append("```")
        lines.append("")

        file_path.write_text("\n".join(lines), encoding="utf-8")

        summary.append({
            "index": idx,
            "stage_name": stage_name,
            "success": success,
            "error": error_text,
            "metadata": metadata,
        })

    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


async def _run_pipeline(
    pipeline_factory: Callable[..., "Pipeline"],
    query_path: Path,
    use_stub: bool,
    output_path: Path,
    pipeline_kwargs: Optional[Dict] = None,
) -> Path:
    from pipeline import Pipeline

    pipeline_kwargs = pipeline_kwargs or {}

    print(f"[info] building pipeline '{pipeline_factory.__name__}' for query '{query_path}'")

    question_text = None
    try:
        relative_query = query_path.relative_to(USER_QUERY_DIR)
        print(f"[info] loading user query template '{relative_query}'")
        # Check if pipeline factory accepts user_query_template parameter
        import inspect
        sig = inspect.signature(pipeline_factory)
        if 'user_query_template' in sig.parameters:
            pipeline = pipeline_factory(user_query_template=str(relative_query), **pipeline_kwargs)
        else:
            pipeline = pipeline_factory(**pipeline_kwargs)
            question_text = load_user_query(str(relative_query))
            pipeline.set_context({"user_query": question_text})
        pipeline_input = None
    except ValueError:
        question_text = query_path.read_text(encoding="utf-8")
        pipeline = pipeline_factory(**pipeline_kwargs)
        pipeline.set_context({"user_query": question_text})
        pipeline_input = question_text
        print("[info] loaded external query file; injected text into pipeline context")

    print(f"[info] executing pipeline '{pipeline.name}' with {len(pipeline.stages)} stages…")
    def on_progress(event: str, payload: Dict[str, Any]) -> None:
        name = payload.get("stage_name")
        idx = payload.get("stage_index")
        total = payload.get("stage_total")
        if event == "stage_start":
            print(f"[stage {idx}/{total}] start: {name}")
        elif event == "stage_complete":
            duration = payload.get("duration_sec")
            print(f"[stage {idx}/{total}] complete: {name} ({duration}s)")
        elif event in {"stage_error", "stage_failed"}:
            print(f"[stage {idx}/{total}] {event}: {name} -> {payload.get('error')}")

    result = await pipeline.execute(pipeline_input, None, progress_callback=on_progress)
    if not result.success:
        print("[error] pipeline execution failed; see logs/pipeline_trace.md for details")
        raise RuntimeError(f"Pipeline execution failed: {result.error}")

    response_text = str(result.data or "")

    if question_text is None:
        question_text = query_path.read_text(encoding="utf-8")
    question_section = ["# Question", question_text.strip(), ""]
    response_section = [f"# {pipeline.name.replace('_', ' ').title()} Response", response_text.strip()]

    output_path.write_text("\n".join(question_section + ["---", ""] + response_section) + "\n", encoding="utf-8")
    print(f"[info] wrote combined question/response to {output_path}")

    work_dir = output_path.parent / output_path.stem
    _write_stage_artifacts(result.stage_results or [], work_dir)
    if work_dir.exists():
        print(f"[info] wrote stage artifacts to {work_dir}")

    # If any stages failed, summarize and optionally fail the run
    failed_stages = []
    try:
        for idx, sr in enumerate(result.stage_results or [], start=1):
            if not getattr(sr, "success", True):
                stage_name = (getattr(sr, "metadata", {}) or {}).get("agent_name") or f"stage_{idx}"
                failed_stages.append((stage_name, getattr(sr, "error", "Unknown error")))
    except Exception:
        pass

    if failed_stages:
        print(f"[warn] {len(failed_stages)} stage(s) failed during execution:")
        for name, err in failed_stages:
            print(f"       - {name}: {err}")

        # Detect likely authentication errors to provide a clearer message and non-zero exit
        def _is_auth_error(msg: str) -> bool:
            low = (msg or "").lower()
            return (
                "invalid api key" in low
                or "incorrect api key" in low
                or "authentication" in low
                or "401" in low
            )

        if any(_is_auth_error(err or "") for _, err in failed_stages):
            print("[error] Authentication failed (invalid or missing API key). Check OPENAI_API_KEY and any search keys (SERPER_API_KEY) in your .env. See docs/ENVIRONMENT.md.")
            raise RuntimeError("Authentication error: invalid or missing API key(s)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline against a Markdown query template and save the response.")
    parser.add_argument("pipeline", choices=["prediction", "prediction_stub", "research"], help="Pipeline to execute.")
    parser.add_argument("query", help="Path or name of the Markdown/text file containing the user query.")
    parser.add_argument("--output", help="Optional output Markdown file for the response.")
    parser.add_argument("--stub", action="store_true", help="Use stubbed pipeline where available (prediction only).")

    args = parser.parse_args()

    pipeline_factories: Dict[str, Callable[..., "Pipeline"]] = {
        "prediction": create_prediction_pipeline,
        "prediction_stub": create_prediction_pipeline_stub,
        "research": create_research_pipeline,
    }

    pipeline_key = args.pipeline
    if args.stub and pipeline_key == "prediction":
        pipeline_key = "prediction_stub"

    factory = pipeline_factories[pipeline_key]
    query_path = _resolve_query_path(args.query)
    output_path = _derive_response_path(query_path, pipeline_key, args.output)

    print(f"[info] using output file {output_path}")

    try:
        asyncio.run(_run_pipeline(factory, query_path, args.stub, output_path))
        print("[done] pipeline run complete")
    except Exception as e:
        # Fail gracefully with a clear message and non-zero exit without stack trace noise
        print(f"[fatal] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

