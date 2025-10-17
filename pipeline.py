from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import asyncio
import logging
import os
import json
from datetime import datetime
from agent import Agent, AgentResult
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    success: bool
    data: Any
    stage_results: List[AgentResult]
    error: Optional[str] = None
    execution_time: Optional[float] = None

class Pipeline:
    """Agent pipeline that executes stages sequentially"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.stages: List[Agent] = []
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.context: Dict[str, Any] = {}
    
    def add_stage(self, agent: Agent) -> 'Pipeline':
        """Add a stage to the pipeline"""
        self.stages.append(agent)
        self.logger.info(f"Added stage: {agent.name}")
        return self
    
    def add_stages(self, agents: List[Agent]) -> 'Pipeline':
        """Add multiple stages to the pipeline"""
        for agent in agents:
            self.add_stage(agent)
        return self
    
    def set_context(self, context: Dict[str, Any]) -> 'Pipeline':
        """Set pipeline context that will be passed to all stages"""
        self.context.update(context)
        return self
    
    def update_context(self, key: str, value: Any) -> 'Pipeline':
        """Update a specific context value"""
        self.context[key] = value
        return self
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Execute the pipeline with all stages in sequence"""
        start_time = asyncio.get_event_loop().time()
        
        # Merge provided context with pipeline context
        execution_context = {**self.context}
        if context:
            execution_context.update(context)
        # Initialize previous_response_id for potential cross-stage chaining
        execution_context.setdefault("previous_response_id", None)
        
        stage_results = []
        default_input = execution_context.get("user_query") or execution_context.get("default_input")
        original_input = input_data if input_data is not None else default_input
        current_data = None
        # Stage 1 multi-draft expansion payload
        stage_one_bundle: Optional[Dict[str, Any]] = None

        # Control file-based trace logging via configuration
        file_logging_enabled = bool(getattr(Config, 'FILE_LOGGING_ENABLED', True))
        stage_log_path = os.path.join('logs', 'stage_responses.jsonl')
        md_log_path = os.path.join('logs', 'pipeline_trace.md')

        header_written = False

        def _append_stage_event(event: Dict[str, Any]) -> None:
            if not file_logging_enabled:
                return
            try:
                # Ensure logs directory exists only if we're writing
                try:
                    os.makedirs('logs', exist_ok=True)
                except Exception:
                    pass
                # Ensure JSON serializable
                safe_event: Dict[str, Any] = {}
                for k, v in event.items():
                    try:
                        json.dumps(v, default=str)
                        safe_event[k] = v
                    except Exception:
                        safe_event[k] = str(v)
                with open(stage_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(safe_event, ensure_ascii=False, default=str, indent=2) + "\n\n")
                # Also append a readable Markdown trace for quick inspection
                try:
                    with open(md_log_path, 'a', encoding='utf-8') as mf:
                        nonlocal header_written
                        if event.get('event') == 'start':
                            if not header_written:
                                header_written = True
                                mf.write(f"## Pipeline Trace: {event.get('pipeline')}\n")
                                mf.write(f"- **run_started**: {event.get('timestamp')}\n")
                                mf.write(f"- **stages**: {event.get('stage_total')}\n\n")
                            mf.write(f"\n### Stage {event.get('stage_index')}/{event.get('stage_total')}: {event.get('stage_name')} — START ({event.get('timestamp')})\n\n")
                            mf.write(f"- **input**:\n\n```text\n")
                            mf.write(str(event.get('input_full') or event.get('input_preview') or ''))
                            mf.write("\n```\n")
                        elif event.get('event') == 'complete':
                            meta = event.get('metadata') or {}
                            mf.write(f"\n### Stage {event.get('stage_index')}/{event.get('stage_total')}: {event.get('stage_name')} — COMPLETE ({event.get('duration_sec')}s)\n\n")
                            if meta.get('model'):
                                mf.write(f"- **model**: {meta.get('model')}  \n")
                            if meta.get('reasoning_effort'):
                                mf.write(f"- **reasoning_effort**: {meta.get('reasoning_effort')}  \n")
                            if 'request' in meta and meta['request']:
                                mf.write("\n- **request sent to OpenAI**:\n\n```json\n")
                                mf.write(json.dumps(meta['request'], indent=2, ensure_ascii=False, default=str))
                                mf.write("\n```\n")
                            preview = event.get('output_full') or event.get('output_preview') or ''
                            mf.write("\n- **stage output**:\n\n")
                            mf.write(f"  - **length**: {event.get('data_len')} chars\n\n")
                            if preview:
                                mf.write("```text\n")
                                mf.write(str(preview))
                                mf.write("\n```\n")
                        elif event.get('event') in ('failed', 'error'):
                            mf.write(f"\n### Stage {event.get('stage_index')}: {event.get('stage_name')} — {event.get('event').upper()}\n\n")
                            mf.write(f"- **error**: {event.get('error')}  \n")
                            mf.write(f"- **duration**: {event.get('duration_sec')}s  \n")
                except Exception:
                    pass
            except Exception:
                # Never fail pipeline because of logging
                pass
        
        def _emit(event: str, payload: Dict[str, Any]) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(event, payload)
            except Exception:
                self.logger.debug("progress callback raised", exc_info=True)

        try:
            _emit("pipeline_start", {
                "pipeline": self.name,
                "stage_total": len(self.stages),
            })
            self.logger.info(f"Starting pipeline execution with {len(self.stages)} stages")
            if Config.DEBUG:
                self.logger.info(f"[DEBUG] Pipeline '{self.name}' context: {execution_context}")
            
            for i, stage in enumerate(self.stages):
                stage_index = i + 1
                stage_total = len(self.stages)
                self.logger.info(f"Executing stage {stage_index}/{stage_total}: {stage.name}")

                # Determine input to this stage: first stage uses original input; subsequent use previous data if available
                if i == 0:
                    stage_input = original_input
                elif stage_one_bundle is not None:
                    stage_input = {
                        "stage_1": stage_one_bundle,
                        "original_input": original_input,
                        "previous_stage_output": current_data,
                    }
                else:
                    stage_input = current_data if current_data is not None else original_input

                if Config.DEBUG:
                    try:
                        preview_val = stage_input
                        if isinstance(stage_input, dict) and "stage_1" in stage_input:
                            preview_val = {
                                **stage_input,
                                "stage_1": {
                                    k: (v if k != "drafts" else f"<list len={len(v) if isinstance(v, list) else 0}>")
                                    for k, v in stage_input["stage_1"].items()
                                },
                            }
                        self.logger.info(
                            f"[DEBUG] Input to stage '{stage.name}' ({len(str(preview_val))} chars):\n{str(preview_val)}"
                        )
                    except Exception:
                        self.logger.info(
                            f"[DEBUG] Input to stage '{stage.name}' ({len(str(stage_input))} chars)"
                        )

                # Mark stage start
                stage_start = asyncio.get_event_loop().time()
                try:
                    log_input = stage_input
                    if isinstance(stage_input, dict) and "stage_1" in stage_input:
                        redacted_stage1 = {}
                        for k, v in stage_input["stage_1"].items():
                            if k == "drafts" and isinstance(v, list):
                                redacted_stage1[k] = [f"<draft len={len(str(d))}>" for d in v]
                            else:
                                redacted_stage1[k] = v
                        log_input = dict(stage_input)
                        log_input["stage_1"] = redacted_stage1
                except Exception:
                    log_input = stage_input

                event_payload = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "pipeline": self.name,
                    "stage_index": stage_index,
                    "stage_total": stage_total,
                    "stage_name": stage.name,
                    "event": "start",
                    "input_preview": str(log_input)[: Config.TRACE_PREVIEW_CHARS if hasattr(Config, 'TRACE_PREVIEW_CHARS') else 512],
                    "input_full": str(log_input),
                }
                _append_stage_event(event_payload)
                _emit("stage_start", event_payload)

                # Execute stage (ensure no cross-stage chaining)
                try:
                    # Execute stage with any available previous_response_id for chaining
                    result = await stage.execute(stage_input, execution_context)
                except BaseException as e:
                    # Handle cancellation/timeouts (e.g., anyio.CancelledError, asyncio.TimeoutError)
                    err_type = e.__class__.__name__
                    cancelled = err_type in ("CancelledError", "TimeoutError", "TimeoutCancellationError")
                    err_text = str(e) or err_type
                    error_msg = (
                        f"Stage '{stage.name}' timed out or was cancelled: {err_text}" if cancelled else f"Stage '{stage.name}' failed with fatal error [{err_type}]: {err_text}"
                    )
                    self.logger.error(error_msg)
                    duration = asyncio.get_event_loop().time() - stage_start
                    event_payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "pipeline": self.name,
                        "stage_index": stage_index,
                        "stage_total": stage_total,
                        "stage_name": stage.name,
                        "event": "error",
                        "error": f"{err_type}: {str(e)}",
                        "duration_sec": round(duration, 3),
                    }
                    _append_stage_event(event_payload)
                    _emit("stage_error", event_payload)
                    # Continue to next stage if configured
                    if getattr(Config, 'CONTINUE_ON_STAGE_ERROR', True):
                        # record a synthetic failed AgentResult for this stage
                        stage_results.append(AgentResult(success=False, data=None, error=error_msg, metadata={"exception": err_type}))
                        # Do not update current_data; leave as last successful data
                        continue
                    return PipelineResult(
                        success=False,
                        data=current_data,
                        stage_results=stage_results,
                        error=error_msg,
                        execution_time=asyncio.get_event_loop().time() - start_time
                    )
                stage_results.append(result)
                
                if not result.success:
                    error_msg = f"Stage '{stage.name}' failed: {result.error}"
                    self.logger.error(error_msg)
                    if Config.DEBUG:
                        self.logger.info(f"[DEBUG] Stage '{stage.name}' result metadata: {result.metadata}")
                    duration = asyncio.get_event_loop().time() - stage_start
                    event_payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "pipeline": self.name,
                        "stage_index": stage_index,
                        "stage_total": stage_total,
                        "stage_name": stage.name,
                        "event": "failed",
                        "error": result.error,
                        "duration_sec": round(duration, 3),
                        "metadata": result.metadata or {},
                    }
                    _append_stage_event(event_payload)
                    _emit("stage_failed", event_payload)
                    if getattr(Config, 'CONTINUE_ON_STAGE_ERROR', True):
                        # Do not stop the pipeline; proceed to next stage
                        # Keep current_data unchanged so subsequent stages can still use prior successful output
                        continue
                    return PipelineResult(
                        success=False,
                        data=current_data,
                        stage_results=stage_results,
                        error=error_msg,
                        execution_time=asyncio.get_event_loop().time() - start_time
                    )
                
                # Capture and persist previous_response_id for the next stage, if provided
                try:
                    if result.metadata and result.metadata.get("previous_response_id"):
                        execution_context["previous_response_id"] = result.metadata["previous_response_id"]
                        try:
                            if Config.DEBUG:
                                logger.info(f"[DEBUG] Captured previous_response_id for next stage: {execution_context['previous_response_id']}")
                        except Exception:
                            pass
                except Exception:
                    pass
                # Update data for next stage
                current_data = result.data
                if i == 0:
                    if isinstance(result.data, dict):
                        stage_one_bundle = {
                            "drafts": result.data.get("drafts"),
                            "combined": result.data.get("combined"),
                            "draft_count": result.data.get("draft_count"),
                            "raw": result.data,
                            "metadata": result.metadata,
                        }
                    else:
                        stage_one_bundle = {
                            "drafts": [result.data],
                            "combined": result.data,
                            "draft_count": 1,
                            "raw": result.data,
                            "metadata": result.metadata,
                        }
                elif i == 1 and stage_one_bundle is not None:
                    stage_one_bundle["evidence_pack"] = result.data
                    stage_one_bundle.setdefault("metadata", {})
                    if result.metadata:
                        stage_one_bundle["metadata"]["evidence_builder"] = result.metadata
                
                # Update context with stage metadata if available
                if result.metadata:
                    execution_context[f"{stage.name}_metadata"] = result.metadata
                
                duration = asyncio.get_event_loop().time() - stage_start
                self.logger.info(f"Stage '{stage.name}' completed successfully in {duration:.2f}s")
                output_str = str(result.data or "")
                output_len = len(output_str)
                output_preview = output_str[: (Config.TRACE_PREVIEW_CHARS if hasattr(Config, 'TRACE_PREVIEW_CHARS') else 512)]
                event_payload = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "pipeline": self.name,
                    "stage_index": stage_index,
                    "stage_total": stage_total,
                    "stage_name": stage.name,
                    "event": "complete",
                    "duration_sec": round(duration, 3),
                    "data_len": output_len,
                    "metadata": result.metadata or {},
                    "output_preview": output_preview,
                    "output_full": output_str,
                }
                _append_stage_event(event_payload)
                _emit("stage_complete", event_payload)
                try:
                    verbosity = getattr(Config, 'LOG_VERBOSITY', 'info').lower()
                    if getattr(Config, 'ERRORS_ONLY', False):
                        pass
                    elif verbosity == 'verbose':
                        self.logger.info(
                            f"[STAGE_OUTPUT] {stage.name}: ({output_len} chars)\n{output_preview}{'…' if output_len > len(output_preview) else ''}"
                        )
                    elif verbosity == 'info':
                        self.logger.info(
                            f"[STAGE_OUTPUT] {stage.name}: {output_len} chars"
                        )
                    # minimal: don't emit content previews here
                except Exception:
                    pass
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.info(f"Pipeline execution completed in {execution_time:.2f}s")
            _emit("pipeline_complete", {
                "pipeline": self.name,
                "stage_total": len(self.stages),
                "execution_time": execution_time,
                "success": True,
            })
            
            return PipelineResult(
                success=True,
                data=current_data,
                stage_results=stage_results,
                execution_time=execution_time
            )
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            if Config.DEBUG:
                self.logger.exception("[DEBUG] Pipeline execution exception")
            _emit("pipeline_complete", {
                "pipeline": self.name,
                "stage_total": len(self.stages),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "success": False,
                "error": error_msg,
            })
            return PipelineResult(
                success=False,
                data=current_data,
                stage_results=stage_results,
                error=error_msg,
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    def get_stage_names(self) -> List[str]:
        """Get list of stage names in order"""
        return [stage.name for stage in self.stages]
    
    def __str__(self) -> str:
        return f"Pipeline({self.name}, stages={len(self.stages)})"
