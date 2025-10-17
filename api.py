from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import logging
from pipeline import Pipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# Global pipeline registry
pipelines: Dict[str, Pipeline] = {}

# Startup initialization to register default pipelines
@app.on_event("startup")
async def _init_default_pipelines() -> None:
    try:
        from agent_pipeline_declarations import create_prediction_pipeline, create_research_pipeline
        prediction = create_prediction_pipeline()
        pipelines[prediction.name] = prediction
        pipelines.setdefault("prediction_pipeline", prediction)

        research = create_research_pipeline()
        pipelines[research.name] = research

        logger.info(f"Initialized default pipelines: {list(pipelines.keys())}")
    except Exception as e:
        logger.error(f"Failed to initialize default pipelines: {e}")

# Pydantic models for API
class ResponseCreateRequest(BaseModel):
    model: Optional[str] = None
    input: Any
    pipeline_name: Optional[str] = None

class PipelineResponse(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    stage_results: Optional[List[Dict[str, Any]]] = None

class CreatePipelineRequest(BaseModel):
    name: str = Field(..., description="Name of the pipeline")
    description: Optional[str] = Field("", description="Description of the pipeline")
    stages: List[Dict[str, Any]] = Field(..., description="List of stage configurations")

class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ReasoningConfigRequest(BaseModel):
    reasoning_effort: Optional[str] = None  # "low", "medium", "high", "maximum"
    temperature: Optional[float] = None  # None -> use server default
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Union[str, ToolConfig]]] = None  # built-in tool strings or custom function tools
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # e.g., "auto"
    verbosity: Optional[str] = None  # optional application-level control
    text_format_type: Optional[str] = None  # "text" or "json_schema"
    text_json_schema: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None
    max_tool_calls: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    extra_request_kwargs: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None

class StageConfig(BaseModel):
    name: str
    type: str = "gpt"  # Currently only supports GPT agents
    system_prompt: str
    model: str = "gpt-5"
    reasoning_config: Optional[ReasoningConfigRequest] = None

# -------- Minimal OpenAI-compatible endpoint (/v1/chat/completions) --------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    pipeline_name: Optional[str] = Config.DEFAULT_PIPELINE_NAME
    stream: Optional[bool] = False
    diagnostics: Optional[bool] = False

# -------- Simple SSE test endpoint (no model calls) --------
class StreamTestRequest(BaseModel):
    duration_sec: Optional[int] = 360  # default 6 minutes

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, authorization: Optional[str] = Header(default=None)):
    # Require Bearer auth only when a server API key is configured
    if Config.SERVER_AUTH_ENABLED:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != Config.SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Join all user/assistant content as input; prefer latest user message
    user_text = "\n".join(m.content for m in req.messages if m.role == "user") or (req.messages[-1].content if req.messages else "")
    pipeline_name = req.pipeline_name or Config.DEFAULT_PIPELINE_NAME
    pipeline = pipelines.get(pipeline_name)
    if not pipeline:
        raise HTTPException(status_code=503, detail="pipeline not ready")

    if req.stream:
        async def event_stream():
            request_id = f"cmpl-local-{int(time.time()*1000)}"
            model_id = req.model or "gpt-5"
            # Initial role chunk
            first = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            # Comment keep-alive to force flush through intermediaries
            yield ": start\n\n"
            await asyncio.sleep(0)

            task = asyncio.create_task(pipeline.execute(user_text, None))
            # Heartbeat every ~30s to keep long streams alive when desired
            heartbeat_interval_seconds = 30
            diagnostics_enabled = bool(req.diagnostics)
            hb_count = 0
            start_ts = time.time()
            while not task.done():
                await asyncio.sleep(heartbeat_interval_seconds)
                hb_count += 1
                hb = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(hb)}\n\n"
                # Extra comment heartbeat to defeat proxy buffering
                yield ": heartbeat\n\n"
                if diagnostics_enabled:
                    diag = {
                        "ts": int(time.time()),
                        "elapsed_sec": int(time.time() - start_ts),
                        "heartbeats": hb_count,
                        "status": "running"
                    }
                    yield f": diag {json.dumps(diag)}\n\n"
                await asyncio.sleep(0)

            result = await task
            if not result.success:
                # Emit final error info, then DONE
                try:
                    stage_summaries = []
                    for i, sr in enumerate(result.stage_results or []):
                        meta = sr.metadata or {}
                        stage_summaries.append({
                            "index": i + 1,
                            "name": meta.get("agent_name"),
                            "success": bool(sr.success),
                            "error": sr.error,
                        })
                    debug_text = f"Pipeline failed: {result.error or 'unknown'}\n\nStages: {json.dumps(stage_summaries)}"
                except Exception:
                    debug_text = f"Pipeline failed: {result.error or 'unknown'}"
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": debug_text}, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            text = str(result.data or "")
            out = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(out)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    # Non-stream path
    result = await pipeline.execute(user_text, None)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "pipeline_failed")
    text = str(result.data or "")
    return {
        "id": "cmpl-local-1",
        "object": "chat.completion",
        "created": 0,
        "model": req.model or "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ]
    }

@app.post("/v1/chat/completions_nohb")
async def chat_completions_no_heartbeat(req: ChatCompletionsRequest, authorization: Optional[str] = Header(default=None)):
    # Require Bearer auth only when a server API key is configured
    if Config.SERVER_AUTH_ENABLED:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != Config.SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Join all user/assistant content as input; prefer latest user message
    user_text = "\n".join(m.content for m in req.messages if m.role == "user") or (req.messages[-1].content if req.messages else "")
    pipeline_name = req.pipeline_name or Config.DEFAULT_PIPELINE_NAME
    pipeline = pipelines.get(pipeline_name)
    if not pipeline:
        raise HTTPException(status_code=503, detail="pipeline not ready")

    if req.stream:
        async def event_stream():
            request_id = f"cmpl-local-{int(time.time()*1000)}"
            model_id = req.model or "gpt-5"
            # Initial role chunk
            first = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            # Optional single comment to help initial flush; no periodic heartbeats
            yield ": start\n\n"
            await asyncio.sleep(0)

            result = await pipeline.execute(user_text, None)
            if not result.success:
                try:
                    stage_summaries = []
                    for i, sr in enumerate(result.stage_results or []):
                        meta = sr.metadata or {}
                        stage_summaries.append({
                            "index": i + 1,
                            "name": meta.get("agent_name"),
                            "success": bool(sr.success),
                            "error": sr.error,
                        })
                    debug_text = f"Pipeline failed: {result.error or 'unknown'}\n\nStages: {json.dumps(stage_summaries)}"
                except Exception:
                    debug_text = f"Pipeline failed: {result.error or 'unknown'}"
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": debug_text}, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            text = str(result.data or "")
            out = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(out)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    # Non-stream path
    result = await pipeline.execute(user_text, None)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "pipeline_failed")
    text = str(result.data or "")
    return {
        "id": "cmpl-local-1",
        "object": "chat.completion",
        "created": 0,
        "model": req.model or "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ]
    }

@app.post("/v1/stream/test/asgi")
async def stream_test_asgi(req: StreamTestRequest, authorization: Optional[str] = Header(default=None)):
    # Require Bearer auth to match other endpoints when enabled
    if Config.SERVER_AUTH_ENABLED:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != Config.SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def event_stream():
        request_id = f"test-{int(time.time()*1000)}"
        model_id = "stream-test"
        # Initial chunk ASAP
        first = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        # Heartbeats for duration
        remaining = max(1, int(req.duration_sec or 360))
        start = time.time()
        while time.time() - start < remaining:
            await asyncio.sleep(5)
            hb = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(hb)}\n\n"

        # Final content
        out = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": "stream-test-complete"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(out)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Content-Encoding": "identity",
        "X-Accel-Buffering": "no",
    })

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agent Framework API",
        "version": "1.0.0",
        "available_pipelines": list(pipelines.keys())
    }

@app.get("/pipelines")
async def list_pipelines():
    """List all available pipelines"""
    payload = {
        "pipelines": [
            {
                "name": name,
                "description": pipeline.description,
                "stages": pipeline.get_stage_names()
            }
            for name, pipeline in pipelines.items()
        ]
    }
    if Config.DEBUG:
        payload["debug"] = {"count": len(pipelines)}
    return payload

@app.post("/pipelines")
async def create_pipeline(request: CreatePipelineRequest):
    """Create a new pipeline"""
    try:
        from agent import GPTAgent, ReasoningConfig, ToolDefinition
        
        # Create pipeline
        pipeline = Pipeline(request.name, request.description or "")
        
        # Add stages
        for stage_config in request.stages:
            if stage_config["type"] == "gpt":
                # Create reasoning config if provided
                reasoning_config = None
                if "reasoning_config" in stage_config and stage_config["reasoning_config"]:
                    rc_data = stage_config["reasoning_config"]
                    
                    # Convert tool configs: allow built-in tool strings or function tools
                    tools: List[Union[str, ToolDefinition]] = []
                    if rc_data.get("tools"):
                        for t in rc_data["tools"]:
                            if isinstance(t, str):
                                tools.append(t)
                            else:
                                tool = ToolDefinition(
                                    name=t["name"],
                                    description=t["description"],
                                    parameters=t["parameters"]
                                )
                                tools.append(tool)
                    
                    reasoning_config = ReasoningConfig(
                        reasoning_effort=rc_data.get("reasoning_effort"),
                        temperature=rc_data.get("temperature"),
                        max_output_tokens=rc_data.get("max_output_tokens"),
                        top_p=rc_data.get("top_p"),
                        stop=rc_data.get("stop"),
                        tools=tools,
                        tool_choice=rc_data.get("tool_choice"),
                        verbosity=rc_data.get("verbosity"),
                        text_format_type=rc_data.get("text_format_type"),
                        text_json_schema=rc_data.get("text_json_schema"),
                        include=rc_data.get("include"),
                        max_tool_calls=rc_data.get("max_tool_calls"),
                        parallel_tool_calls=rc_data.get("parallel_tool_calls"),
                        extra_request_kwargs=rc_data.get("extra_request_kwargs"),
                        timeout_seconds=rc_data.get("timeout_seconds")
                    )
                
                agent = GPTAgent(
                    name=stage_config["name"],
                    system_prompt=stage_config["system_prompt"],
                    model=stage_config.get("model", "gpt-5"),
                    reasoning_config=reasoning_config
                )
                pipeline.add_stage(agent)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported stage type: {stage_config['type']}"
                )
        
        # Register pipeline
        pipelines[request.name] = pipeline
        
        return {
            "message": f"Pipeline '{request.name}' created successfully",
            "pipeline": {
                "name": request.name,
                "description": request.description,
                "stages": pipeline.get_stage_names()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/responses")
async def responses_create(request: ResponseCreateRequest):
    """Create a response. If pipeline_name is supplied, execute that pipeline. Minimal surface."""
    try:
        if not request.pipeline_name:
            raise HTTPException(status_code=400, detail="pipeline_name is required")
        if request.pipeline_name not in pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline '{request.pipeline_name}' not found")
        pipeline = pipelines[request.pipeline_name]
        result = await pipeline.execute(request.input, None)
        response_body = {
            "object": "response",
            "status": "completed" if result.success else "failed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": str(result.data or "")}]
            }]
        }
        try:
            logger.info(f"[CLIENT] Response body: {response_body}")
        except Exception:
            pass
        return response_body
        
        # Convert stage results to dict format
        stage_results = []
        if result.stage_results:
            for stage_result in result.stage_results:
                stage_results.append({
                    "success": stage_result.success,
                    "data": stage_result.data,
                    "error": stage_result.error,
                    "metadata": stage_result.metadata
                })
        
        resp = PipelineResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time=result.execution_time,
            stage_results=stage_results
        )
        if Config.DEBUG:
            logger.info(f"[DEBUG] Execute payload stages={len(stage_results)} success={result.success}")
        return resp
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipelines/{pipeline_name}")
async def get_pipeline(pipeline_name: str):
    """Get pipeline details"""
    if pipeline_name not in pipelines:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_name}' not found"
        )
    
    pipeline = pipelines[pipeline_name]
    payload = {
        "name": pipeline.name,
        "description": pipeline.description,
        "stages": pipeline.get_stage_names()
    }
    if Config.DEBUG:
        payload["debug"] = True
    return payload

@app.delete("/pipelines/{pipeline_name}")
async def delete_pipeline(pipeline_name: str):
    """Delete a pipeline"""
    if pipeline_name not in pipelines:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_name}' not found"
        )
    
    del pipelines[pipeline_name]
    return {"message": f"Pipeline '{pipeline_name}' deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    from config import Config
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
