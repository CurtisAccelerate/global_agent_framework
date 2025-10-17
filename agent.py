from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import json
import time
import os
from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class AgentResult:
    """Result from an agent stage execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

@dataclass
class ToolDefinition:
    """Definition of a tool/function that can be called by the agent"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None

@dataclass
class ReasoningConfig:
    """Configuration for reasoning effort and advanced model parameters (Responses API)"""
    reasoning_effort: Optional[str] = None  # "low", "medium", "high", "maximum"
    temperature: Optional[float] = None  # None -> use server default
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    # Built-in tools or custom function tools
    tools: Optional[List[Union[str, ToolDefinition, Dict[str, Any]]]] = None  # supports built-in tool strings, dict-form tool specs, or function tools
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # "auto", "required", {"type": ...}
    verbosity: Optional[str] = None  # not official; if set, passed via metadata or instructions
    max_tool_calls: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    include: Optional[List[str]] = None
    # Extra request kwargs to pass directly to Responses API (e.g., include, service_tier)
    extra_request_kwargs: Optional[Dict[str, Any]] = None
    # Text output formatting
    text_format_type: Optional[str] = None  # "text" (default) or "json_schema"
    text_json_schema: Optional[Dict[str, Any]] = None  # required when text_format_type=="json_schema"
    # Per-stage timeout control (seconds) for the Responses API call
    timeout_seconds: Optional[int] = None

class Agent(ABC):
    """Base class for all agents in the pipeline"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agent.{name}")
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent's main logic"""
        pass
    
    def __str__(self) -> str:
        return f"Agent({self.name})"

class GPTAgent(Agent):
    """Agent that uses GPT-5 for processing with advanced reasoning and tool support"""
    
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-5", 
                 reasoning_config: Optional[ReasoningConfig] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.system_prompt = system_prompt
        self.model = model
        self.reasoning_config = reasoning_config or ReasoningConfig()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import AsyncOpenAI
            Config.validate()
            self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            if Config.DEBUG:
                self.logger.info("OpenAI Async client initialized for Responses API")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def add_tool(self, tool: ToolDefinition):
        """Add a tool to the agent's available tools"""
        if not self.reasoning_config.tools:
            self.reasoning_config.tools = []
        self.reasoning_config.tools.append(tool)
        self.logger.info(f"Added tool: {tool.name}")
    
    def add_tools(self, tools: List[ToolDefinition]):
        """Add multiple tools to the agent"""
        for tool in tools:
            self.add_tool(tool)
    
    def set_reasoning_effort(self, effort: str):
        """Set the reasoning effort level"""
        valid_efforts = ["low", "medium", "high", "maximum"]
        if effort not in valid_efforts:
            raise ValueError(f"Invalid reasoning effort. Must be one of: {valid_efforts}")
        self.reasoning_config.reasoning_effort = effort
    
    def set_verbosity(self, verbosity: str):
        """Set the verbosity level"""
        valid_verbosity = ["low", "medium", "high"]
        if verbosity not in valid_verbosity:
            raise ValueError(f"Invalid verbosity. Must be one of: {valid_verbosity}")
        self.reasoning_config.verbosity = verbosity
    
    def _prepare_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        """Convert tool specifications to Responses API format.
        Supports built-in tool strings (e.g., "web_search") and function tools.
        """
        if not self.reasoning_config.tools:
            return None
        tools: List[Dict[str, Any]] = []
        for tool in self.reasoning_config.tools:
            if isinstance(tool, dict):
                # Pass through dict-form tool specifications (e.g., {"type": "web_search_preview_2025_03_11"})
                if tool.get("type"):
                    # Normalize legacy nested function schema to flattened Responses API schema
                    if tool.get("type") == "function" and "name" not in tool and isinstance(tool.get("function"), dict):
                        fn = tool.get("function") or {}
                        # Ensure parameters has required including all property keys (per Responses function schema requirements)
                        params = fn.get("parameters") or {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        if isinstance(params, dict):
                            props = params.get("properties") or {}
                            if isinstance(props, dict):
                                params = dict(params)
                                params["type"] = "object"
                                params["properties"] = props
                                params["additionalProperties"] = params.get("additionalProperties", False)
                                # Force required to include every property key
                                params["required"] = list(props.keys())
                        flattened = {
                            "type": "function",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "parameters": params,
                            "strict": fn.get("strict", True),
                        }
                        tools.append(flattened)
                    else:
                        tools.append(tool)
                else:
                    # Ignore unknown dict tool formats here; they may be function specs handled below
                    pass
            elif isinstance(tool, str):
                # Built-in tool reference per Responses API
                # Use web_search directly (no deprecated mapping needed)
                tools.append({"type": tool})
            else:
                # Local function tool with strict schema when possible
                # Responses API expects flattened function schema at top-level
                # Normalize parameters: ensure required includes all property keys
                params: Dict[str, Any] = tool.parameters or {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                if isinstance(params, dict):
                    props = params.get("properties") or {}
                    if isinstance(props, dict):
                        params = dict(params)
                        params["type"] = "object"
                        params["properties"] = props
                        params["additionalProperties"] = params.get("additionalProperties", False)
                        if "required" not in params:
                            params["required"] = []
                tools.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                    "strict": True,
                })
                tools.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                    "strict": True,
                })
        return tools
    
    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call and return the result"""
        # Support dict-style or SDK object-style tool_call structures
        if isinstance(tool_call, dict):
            function_name = tool_call.get("function", {}).get("name")
            function_args = tool_call.get("function", {}).get("arguments", "{}")
        else:
            function_name = getattr(getattr(tool_call, "function", None), "name", None)
            function_args = getattr(getattr(tool_call, "function", None), "arguments", "{}")
        
        # Find the tool definition
        tool_def = None
        for tool in self.reasoning_config.tools:
            if tool.name == function_name:
                tool_def = tool
                break
        
        if not tool_def or not tool_def.function:
            raise ValueError(f"Tool '{function_name}' not found or has no function")
        
        try:
            # Parse arguments
            args = json.loads(function_args)
            
            # Execute the function (offload sync work to a thread so the event loop stays responsive)
            if asyncio.iscoroutinefunction(tool_def.function):
                result = await tool_def.function(**args)
            else:
                result = await asyncio.to_thread(tool_def.function, **args)
            
            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed for {function_name}: {e}")
            # Optionally continue by returning a structured error object
            continue_on_tool_error = getattr(Config, 'CONTINUE_ON_TOOL_ERROR', True)
            if continue_on_tool_error:
                return {"error": f"{type(e).__name__}: {str(e)}", "tool": function_name}
            raise
    
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute via Responses API with built-in tools and local function tools.

        Implements a minimal multi-turn function-calling loop:
        1) Call the model.
        2) If function calls are returned, execute them locally.
        3) Send their outputs back to the model using previous_response_id.
        4) Repeat once more if the model makes another function call.
        """
        try:
            if not self.client:
                self._initialize_client()
            # Build Responses API request
            # Minimal Responses API: instructions + input (string). Use previous_response_id to chain.
            req: Dict[str, Any] = {
                "model": self.model,
                "input": str(input_data),
            }
            if context and context.get("previous_response_id"):
                req["previous_response_id"] = context["previous_response_id"]
            # System instructions (+ optional verbosity directive)
            # Stage rules -> instructions (developer/system semantics)
            instructions = self.system_prompt or ""
            if self.reasoning_config.verbosity:
                instructions = (instructions + "\n\n" if instructions else "") + (
                    f"Output verbosity: {self.reasoning_config.verbosity}. "
                    "Use concise output for low, balanced detail for medium, and comprehensive detail for high."
                )
            if instructions:
                req["instructions"] = instructions
            # Reasoning config
            reasoning: Dict[str, Any] = {}
            if self.reasoning_config.reasoning_effort:
                reasoning["effort"] = self.reasoning_config.reasoning_effort
            if reasoning:
                req["reasoning"] = reasoning
            # Temperature uses server default if None
            if self.reasoning_config.temperature is not None:
                req["temperature"] = self.reasoning_config.temperature
            # Max output tokens
            if self.reasoning_config.max_output_tokens is not None:
                req["max_output_tokens"] = self.reasoning_config.max_output_tokens
            # top_p
            if self.reasoning_config.top_p is not None:
                req["top_p"] = self.reasoning_config.top_p
            # Structured outputs (Stage 3): json_schema
            fmt_type = (self.reasoning_config.text_format_type or "text").lower()
            if fmt_type == "json_schema" and self.reasoning_config.text_json_schema:
                req["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": f"{self.name}_schema",
                        "json_schema": self.reasoning_config.text_json_schema,
                        "strict": True
                    }
                }
            # Tools
            tools = self._prepare_tools_for_api()
            if tools:
                req["tools"] = tools
                if self.reasoning_config.tool_choice:
                    req["tool_choice"] = self.reasoning_config.tool_choice
            elif self.reasoning_config.tool_choice:
                # Allow forcing tool_choice even without tools (e.g., "none")
                req["tool_choice"] = self.reasoning_config.tool_choice
            if self.reasoning_config.max_tool_calls is not None:
                req["max_tool_calls"] = self.reasoning_config.max_tool_calls
            if self.reasoning_config.parallel_tool_calls is not None:
                req["parallel_tool_calls"] = self.reasoning_config.parallel_tool_calls
            if self.reasoning_config.include:
                req["include"] = self.reasoning_config.include
            # Merge extra request kwargs carefully (don't clobber explicit req fields)
            if self.reasoning_config.extra_request_kwargs:
                extra = dict(self.reasoning_config.extra_request_kwargs)
                # Merge include lists if both provided
                if "include" in extra:
                    existing_inc = req.get("include")
                    if existing_inc:
                        existing_list = existing_inc if isinstance(existing_inc, list) else [existing_inc]
                        extra_list = extra["include"] if isinstance(extra["include"], list) else [extra["include"]]
                        req["include"] = list(dict.fromkeys(existing_list + extra_list))
                    else:
                        req["include"] = extra["include"]
                    extra.pop("include", None)
                for k, v in extra.items():
                    if k not in req:
                        req[k] = v
            # Avoid attaching application-level context into request to minimize schema issues

            # Call Responses API with timeout
            if Config.DEBUG:
                self.logger.info(f"[DEBUG] Request -> /v1/responses: {json.dumps(req)[:2000]}")
            
            response = None
            did_fallback_without_tools = False
            final_response = None
            aggregated_tool_results: List[Dict[str, Any]] = []
            timeout_seconds = (
                self.reasoning_config.timeout_seconds
                or getattr(Config, 'RESPONSES_TIMEOUT_SECONDS', None)
                or 300
            )
            # New robust loop: iterate until no tool calls (with safety cap), accumulate text, wrap-up if needed
            max_turns = self.reasoning_config.max_tool_calls or 8
            previous_response_id = None
            final_response = None
            tool_outputs_for_model: List[Dict[str, Any]] = []
            accum_text_parts: List[str] = []

            for iteration in range(max_turns):
                # Prepare request for this turn
                if iteration == 0:
                    call_req = dict(req)
                else:
                    call_req = dict(req)
                    call_req["previous_response_id"] = previous_response_id
                    call_req["input"] = [
                        {"type": "function_call_output", "call_id": tr.get("call_id"), "output": tr.get("output")}
                        for tr in tool_outputs_for_model
                    ]

                # Invoke
                try:
                    response = await asyncio.wait_for(
                        self.client.responses.create(**call_req),
                        timeout=timeout_seconds
                    )
                    if Config.DEBUG:
                        self.logger.info(f"[DEBUG] API call (turn {iteration+1}) completed successfully")
                except asyncio.TimeoutError as e:
                    self.logger.error("Stage timed out: %s", type(e).__name__)
                    self.logger.exception("Stage timeout traceback")
                    retry_without_tools = (iteration == 0) and (bool(req.get("tools")) or (req.get("tool_choice") not in (None, "none")))
                    if retry_without_tools:
                        try:
                            if Config.DEBUG:
                                self.logger.info("[DEBUG] Retrying once without tools due to timeout")
                            req_no_tools = dict(req)
                            req_no_tools.pop("tools", None)
                            req_no_tools["tool_choice"] = "none"
                            response = await asyncio.wait_for(
                                self.client.responses.create(**req_no_tools),
                                timeout=min(timeout_seconds, 120)
                            )
                            did_fallback_without_tools = True
                        except Exception as e2:
                            self.logger.error("Fallback without tools failed: %s %r", type(e2).__name__, e2)
                            self.logger.exception("Fallback failure traceback")
                            raise
                    else:
                        raise
                except asyncio.CancelledError as e:
                    self.logger.error("Stage cancelled: %s", type(e).__name__)
                    self.logger.exception("Stage cancellation traceback")
                    retry_without_tools = (iteration == 0) and (bool(req.get("tools")) or (req.get("tool_choice") not in (None, "none")))
                    if retry_without_tools:
                        try:
                            if Config.DEBUG:
                                self.logger.info("[DEBUG] Retrying once without tools due to cancellation")
                            req_no_tools = dict(req)
                            req_no_tools.pop("tools", None)
                            req_no_tools["tool_choice"] = "none"
                            response = await asyncio.wait_for(
                                self.client.responses.create(**req_no_tools),
                                timeout=min(timeout_seconds, 120)
                            )
                            did_fallback_without_tools = True
                        except Exception as e2:
                            self.logger.error("Fallback without tools after cancellation failed: %s %r", type(e2).__name__, e2)
                            self.logger.exception("Fallback failure traceback")
                            raise Exception(f"API call was cancelled or timed out after {timeout_seconds}s")
                    else:
                        raise Exception(f"API call was cancelled or timed out after {timeout_seconds}s")
                except Exception as e:
                    self.logger.error("Stage failed: %s %r", type(e).__name__, e)
                    self.logger.exception("Stage failure traceback")
                    raise

                output_items = getattr(response, "output", []) or []
                previous_response_id = getattr(response, "id", None)
                if Config.DEBUG:
                    try:
                        self.logger.info(f"[DEBUG] Raw response status={getattr(response,'status',None)} model={getattr(response,'model',None)}")
                        self.logger.info(f"[DEBUG] Response output_items: {len(output_items)} items")
                    except Exception:
                        pass

                tool_calls_meta: List[Dict[str, Any]] = []
                for item in output_items:
                    item_type = getattr(item, "type", None)
                    if item_type == "message":
                        content = getattr(item, "content", []) or []
                        for c in content:
                            if getattr(c, "type", None) == "output_text":
                                txt = getattr(c, "text", "")
                                if txt:
                                    accum_text_parts.append(txt)
                    elif item_type == "reasoning":
                        summary = getattr(item, "summary", None)
                        if summary:
                            accum_text_parts.append(str(summary))
                    if item_type == "function_call":
                        fn = getattr(item, "function", None)
                        call_id_detected = getattr(item, "call_id", None)
                        function_name_detected = getattr(fn, "name", None)
                        function_args_detected = getattr(fn, "arguments", "{}")
                        tool_calls_meta.append({
                            "call_id": call_id_detected,
                            "function": {
                                "name": function_name_detected,
                                "arguments": function_args_detected,
                            },
                        })
                        try:
                            self.logger.info(f"[TOOL] Detected function_call: call_id={call_id_detected} name={function_name_detected}")
                        except Exception:
                            pass

                if not tool_calls_meta:
                    final_response = {
                        "prev_id": previous_response_id,
                        "text": "\n".join([p for p in accum_text_parts if p]).strip(),
                        "model": getattr(response, "model", self.model),
                        "usage": getattr(response, "usage", None),
                    }
                    break

                # Execute tools
                tool_outputs_for_model = []
                for tc in tool_calls_meta:
                    function_name = tc.get("function", {}).get("name")
                    function_args = tc.get("function", {}).get("arguments", "{}")
                    call_id = tc.get("call_id")
                    if not function_name:
                        aggregated_tool_results.append({
                            "tool_call_id": call_id or "unknown",
                            "function_name": "unknown",
                            "error": "Malformed function_call: missing name",
                        })
                        out_err = json.dumps({"error": "malformed function_call: missing name"})
                        tool_outputs_for_model.append({"call_id": call_id, "output": out_err})
                        continue
                    # Find matching local tool
                    tool_def = None
                    for tool in (self.reasoning_config.tools or []):
                        try:
                            if getattr(tool, "name", None) == function_name and hasattr(tool, "function"):
                                tool_def = tool
                                break
                        except Exception:
                            pass
                    if not call_id:
                        raise ValueError("Missing call_id for function_call")
                    if not tool_def or not getattr(tool_def, "function", None):
                        raise ValueError(f"Tool '{function_name}' not found or has no function")
                    # Execute tool
                    try:
                        args = json.loads(function_args or "{}")
                    except Exception:
                        args = {}
                    try:
                        self.logger.info(f"[TOOL] Exec start: call_id={call_id} name={function_name} args={(function_args[:500] if isinstance(function_args, str) else str(function_args)[:500])}")
                        t0 = time.perf_counter()
                        if asyncio.iscoroutinefunction(tool_def.function):
                            result_val = await tool_def.function(**args)
                        else:
                            result_val = await asyncio.to_thread(tool_def.function, **args)
                        try:
                            output_str = json.dumps(result_val)
                        except Exception:
                            output_str = str(result_val)
                        dur_ms = int((time.perf_counter() - t0) * 1000)
                        self.logger.info(f"[TOOL] Exec done: call_id={call_id} name={function_name} duration_ms={dur_ms} output_len={len(output_str)}")
                        preview = (output_str[:800] + "…") if len(output_str) > 800 else output_str
                        self.logger.info(f"[TOOL] Output preview ({function_name}): {preview}")
                        tool_outputs_for_model.append({"call_id": call_id, "output": output_str})
                        aggregated_tool_results.append({"tool_call_id": call_id, "function_name": function_name, "result": result_val})
                    except Exception as e:
                        self.logger.error(f"[TOOL] Exec error: call_id={call_id} name={function_name} err={e}")
                        continue_on_tool_error = getattr(Config, 'CONTINUE_ON_TOOL_ERROR', True)
                        if continue_on_tool_error:
                            # Return structured error back to the model so it can recover
                            err_payload = {"error": f"{type(e).__name__}: {str(e)}", "tool": function_name}
                            try:
                                err_str = json.dumps(err_payload)
                            except Exception:
                                err_str = str(err_payload)
                            tool_outputs_for_model.append({"call_id": call_id, "output": err_str})
                            aggregated_tool_results.append({
                                "tool_call_id": call_id,
                                "function_name": function_name,
                                "error": f"{type(e).__name__}: {str(e)}",
                            })
                            # Continue to next tool call
                            continue
                        # Otherwise, propagate the error to fail the stage
                        raise

            # If exhausted turns without final text, attempt wrap-up forcing text
            if not final_response:
                try:
                    wrap_req = dict(req)
                    wrap_req["previous_response_id"] = previous_response_id
                    wrap_req["tool_choice"] = "none"
                    if tool_outputs_for_model:
                        wrap_req["input"] = [
                            {"type": "function_call_output", "call_id": tr.get("call_id"), "output": tr.get("output")}
                            for tr in tool_outputs_for_model
                        ]
                    wrap_req["instructions"] = (req.get("instructions") or "") + "\n\nREQUIRED: Produce the final textual answer now. Do not call tools."
                    wrap_resp = await asyncio.wait_for(self.client.responses.create(**wrap_req), timeout=min(timeout_seconds, 120))
                    wrap_text_parts: List[str] = []
                    for item in getattr(wrap_resp, "output", []) or []:
                        if getattr(item, "type", None) == "message":
                            for c in getattr(item, "content", []) or []:
                                if getattr(c, "type", None) == "output_text":
                                    wrap_text_parts.append(getattr(c, "text", ""))
                    final_response = {
                        "prev_id": getattr(wrap_resp, "id", previous_response_id),
                        "text": ("\n".join([p for p in wrap_text_parts if p]).strip() or "\n".join([p for p in accum_text_parts if p]).strip()),
                        "model": getattr(wrap_resp, "model", self.model),
                        "usage": getattr(wrap_resp, "usage", None),
                    }
                except Exception:
                    pass

            # Extract output text and previous_response_id for chaining
            # Build final result
            if not final_response:
                # In case the loop did not set final_response (should not happen), attempt to read last response
                prev_id = getattr(response, "id", None)
                result_text_parts: List[str] = []
                for item in getattr(response, "output", []) or []:
                    if getattr(item, "type", None) == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", None) == "output_text":
                                result_text_parts.append(getattr(c, "text", ""))
                final_response = {
                    "prev_id": prev_id,
                    "text": "\n".join([p for p in result_text_parts if p]),
                    "model": getattr(response, "model", self.model),
                    "usage": getattr(response, "usage", None),
                }

            # Do NOT replace assistant text with tool output. If empty, return empty and let caller handle it.
            final_text = final_response.get("text", "") if final_response else ""

            return AgentResult(
                success=True,
                data=final_text,
                tool_calls=aggregated_tool_results or None,
                metadata={
                    "agent_name": self.name,
                    "previous_response_id": final_response.get("prev_id"),
                    "model": final_response.get("model", self.model),
                    "usage": final_response.get("usage"),
                    "reasoning_effort": self.reasoning_config.reasoning_effort,
                    "temperature": self.reasoning_config.temperature,
                    "tools_used": len(aggregated_tool_results) if aggregated_tool_results else 0,
                    "request": req if hasattr(json, 'dumps') and Config.DEBUG else None,
                    "did_fallback_without_tools": did_fallback_without_tools,
                }
            )
            
        except Exception as e:
            try:
                # Build structured error metadata for debugging
                error_meta: Dict[str, Any] = {
                    "error_type": type(e).__name__,
                    "error_repr": repr(e),
                }
                # Include request preview and timeout for context
                try:
                    error_meta["timeout_seconds"] = self.reasoning_config.timeout_seconds or 300
                    if 'req' in locals() and Config.DEBUG:
                        # Avoid logging secrets; the req dict does not include api_key
                        error_meta["request_preview"] = {
                            k: (v if k != "input" else f"{str(v)[:256]}…") for k, v in (req or {}).items()
                        }
                except Exception:
                    pass
                # Attempt to extract HTTP status, request id and headers from the exception
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        error_meta["status_code"] = getattr(resp, "status_code", None) or getattr(resp, "status", None)
                        headers = getattr(resp, "headers", None)
                        if headers:
                            try:
                                # headers may be a dict-like object
                                rid = headers.get("x-request-id") or headers.get("X-Request-Id")
                                error_meta["x_request_id"] = rid
                                error_meta["openai_processing_ms"] = headers.get("openai-processing-ms")
                                error_meta["date"] = headers.get("date")
                            except Exception:
                                pass
                        # Try to capture a small portion of the body
                        try:
                            body_text = None
                            if hasattr(resp, "text") and callable(getattr(resp, "text")):
                                body_text = resp.text
                            elif hasattr(resp, "text"):
                                body_text = resp.text
                            elif hasattr(resp, "json"):
                                body_text = json.dumps(resp.json())
                            if body_text:
                                error_meta["response_body_preview"] = str(body_text)[:500]
                        except Exception:
                            pass
                except Exception:
                    pass
                # Log error with structured context
                self.logger.error("Execution failed: %s", error_meta.get("error_repr", str(e)))
                self.logger.exception("Execution failure traceback with context")
            except Exception:
                self.logger.error(f"Execution failed: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=f"{type(e).__name__}: {str(e)}",
                metadata=error_meta if 'error_meta' in locals() else None
            )


# Optional integration: OpenAI Agents SDK-based agent with built-in tool loop
class AgentsSDKAgent(Agent):
    """Agent implementation that delegates tool-calling loop to OpenAI Agents SDK.

    This keeps our stage code minimal and leverages the SDK's built-in function-calling
    and reconciliation with call_id semantics.
    """

    def __init__(self, name: str, system_prompt: str, model: str = "gpt-5",
                 reasoning_config: Optional[ReasoningConfig] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.system_prompt = system_prompt
        self.model = model
        self.reasoning_config = reasoning_config or ReasoningConfig()

    def add_tool(self, tool: ToolDefinition):
        if not self.reasoning_config.tools:
            self.reasoning_config.tools = []
        self.reasoning_config.tools.append(tool)
        self.logger.info(f"Added tool: {tool.name}")

    def add_tools(self, tools: List[ToolDefinition]):
        for tool in tools:
            self.add_tool(tool)

    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        try:
            # Lazy import to avoid hard dependency if not installed
            try:
                from agents import Agent as OAAgent, Runner, FunctionTool, ModelSettings
                try:
                    from agents import WebSearchTool  # optional hosted tool
                except Exception:
                    WebSearchTool = None
            except Exception as e:
                raise ImportError("openai-agents package is required for AgentsSDKAgent") from e

            # Ensure Agents SDK sees our configured API key
            try:
                # Always force Agents SDK to use Config.OPENAI_API_KEY
                cfg_key = getattr(Config, 'OPENAI_API_KEY', None)
                if cfg_key:
                    os.environ["OPENAI_API_KEY"] = cfg_key
                    try:
                        self.logger.info("[AGENTS] OPENAI_API_KEY set from Config for Agents SDK")
                    except Exception:
                        pass
            except Exception:
                pass

            # Build tool list for SDK
            sdk_tools: List[Any] = []
            tool_use_behavior = None
            # Map built-in tools by string type
            for t in (self.reasoning_config.tools or []):
                if isinstance(t, dict):
                    t_type = t.get("type")
                    if t_type in ("web_search", "web_search_preview") and WebSearchTool is not None:
                        try:
                            sdk_tools.append(WebSearchTool())
                            self.logger.info("[AGENTS] Registered hosted tool: WebSearchTool")
                        except Exception:
                            pass
                elif isinstance(t, str):
                    if t in ("web_search", "web_search_preview") and WebSearchTool is not None:
                        try:
                            sdk_tools.append(WebSearchTool())
                            self.logger.info("[AGENTS] Registered hosted tool: WebSearchTool")
                        except Exception:
                            pass

            # Wrap local function tools
            for t in (self.reasoning_config.tools or []):
                if hasattr(t, "function") and callable(getattr(t, "function", None)):
                    func = t.function
                    params_schema = t.parameters or {"type": "object", "properties": {}, "required": [], "additionalProperties": True}

                    # Capture tool name at definition time to avoid late-binding of 't'
                    tool_name_captured = getattr(t, "name", "unknown")

                    async def on_invoke_tool(ctx, args_json: str, _func=func, _tool_name=tool_name_captured):
                        try:
                            try:
                                args = json.loads(args_json or "{}")
                            except Exception:
                                args = {}
                            try:
                                self.logger.info(f"[TOOL] SDK exec start: name={_tool_name} args={str(args_json)[:500]}")
                            except Exception:
                                pass
                            if asyncio.iscoroutinefunction(_func):
                                result_val = await _func(**args)
                            else:
                                result_val = _func(**args)
                            try:
                                out_str = json.dumps(result_val)
                            except Exception:
                                out_str = str(result_val)
                            try:
                                self.logger.info(f"[TOOL] SDK exec done: name={_tool_name} output_len={len(out_str)}")
                                prev = (out_str[:800] + "…") if len(out_str) > 800 else out_str
                                self.logger.info(f"[TOOL] SDK output preview: {prev}")
                            except Exception:
                                pass
                            return out_str
                        except Exception as e:
                            # Do not kill the SDK run on tool failure; return structured error
                            try:
                                self.logger.error(f"[TOOL] SDK exec error: name={_tool_name} err={e}")
                            except Exception:
                                pass
                            err_payload = {"error": f"{type(e).__name__}: {str(e)}", "tool": _tool_name}
                            try:
                                return json.dumps(err_payload)
                            except Exception:
                                return str(err_payload)

                    try:
                        ft = FunctionTool(
                            name=t.name,
                            description=t.description or t.name,
                            params_json_schema=params_schema,
                            on_invoke_tool=on_invoke_tool,
                        )
                        sdk_tools.append(ft)
                        self.logger.info(f"[AGENTS] Registered function tool: {t.name}")
                    except Exception as e:
                        self.logger.error(f"[AGENTS] Failed to register tool {t.name}: {e}")

            # Model settings
            model_settings = None
            try:
                ms_kwargs: Dict[str, Any] = {}
                if self.reasoning_config.temperature is not None:
                    ms_kwargs["temperature"] = self.reasoning_config.temperature
                if self.reasoning_config.top_p is not None:
                    ms_kwargs["top_p"] = self.reasoning_config.top_p
                if self.reasoning_config.tool_choice is not None:
                    ms_kwargs["tool_choice"] = self.reasoning_config.tool_choice
                if ms_kwargs:
                    model_settings = ModelSettings(**ms_kwargs)
            except Exception:
                pass

            # Build SDK agent (omit tool_use_behavior when None to use SDK default 'run_llm_again')
            oa_kwargs: Dict[str, Any] = {
                "name": self.name,
                "instructions": self.system_prompt,
                "model": self.model,
                "tools": sdk_tools or [],
                "model_settings": model_settings,
            }
            if tool_use_behavior is not None:
                oa_kwargs["tool_use_behavior"] = tool_use_behavior
            oa_agent = OAAgent(**oa_kwargs)

            if Config.DEBUG:
                self.logger.info(f"[AGENTS] Running with tools={len(sdk_tools)}")

            # Execute via Runner with tracing enabled
            from agents import Runner as _Runner
            run_cfg = None
            try:
                from agents.run import RunConfig
                run_cfg = RunConfig(
                    workflow_name=getattr(Config, 'TRACING_WORKFLOW_NAME', 'Agent workflow'),
                    tracing_disabled=bool(int(os.environ.get('OPENAI_AGENTS_DISABLE_TRACING', '0'))),
                    trace_include_sensitive_data=False,
                )
            except Exception:
                pass

            result = await _Runner.run(oa_agent, input=str(input_data or ""), run_config=run_cfg)

            final_text = getattr(result, "final_output", None)
            # If the SDK returned no assistant text, attempt a one-shot wrap-up turn forcing text
            if not final_text:
                try:
                    # Build a clone agent with tool_choice="none" to force textual synthesis
                    wrap_ms = None
                    try:
                        wrap_ms = ModelSettings(tool_choice="none")
                    except Exception:
                        pass
                    wrap_agent = OAAgent(
                        name=self.name,
                        instructions=(self.system_prompt + "\n\nREQUIRED: Produce the final textual answer now. Do not call tools."),
                        model=self.model,
                        tools=sdk_tools or [],
                        model_settings=wrap_ms,
                    )
                    wrap_input = None
                    try:
                        to_input_list = getattr(result, 'to_input_list', None)
                        if callable(to_input_list):
                            wrap_input = to_input_list()
                    except Exception:
                        wrap_input = None
                    if not wrap_input:
                        wrap_input = str(input_data or "")
                    wrap_res = await _Runner.run(wrap_agent, input=wrap_input, run_config=run_cfg)
                    final_text = getattr(wrap_res, 'final_output', None)
                except Exception:
                    final_text = None
            return AgentResult(
                success=bool(final_text),
                data=final_text or "",
                metadata={
                    "agent_name": self.name,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_config.reasoning_effort,
                    "tools_used": None,
                    "agents_sdk": True,
                    "trace_id": getattr(result, 'trace_id', None),
                }
            )
        except Exception as e:
            self.logger.error(f"[AGENTS] Execution failed: {e}")
            return AgentResult(success=False, data=None, error=str(e))


class MultiDraftAgent(Agent):
    """Wrapper that executes a base agent multiple times to produce draft variations."""

    def __init__(
        self,
        base_agent: Agent,
        draft_count: int = 1,
        aggregate_fn: Optional[Callable[[Any, List[Any]], str]] = None,
        name: Optional[str] = None,
    ):
        if not isinstance(base_agent, Agent):
            raise TypeError("base_agent must be an instance of Agent")
        draft_count_int = int(draft_count)
        if draft_count_int < 1:
            draft_count_int = 1
        if draft_count_int > 3:
            draft_count_int = 3

        super().__init__(name=name or base_agent.name, description=base_agent.description)
        self.base_agent = base_agent
        self.draft_count = draft_count_int
        self.aggregate_fn = aggregate_fn or self._default_aggregate

    def _default_aggregate(self, input_data: Any, drafts: List[Any]) -> str:
        if len(drafts) == 1:
            single = drafts[0]
            return "" if single is None else str(single)

        lines: List[str] = []
        lines.append("# Stage 1 Combined Drafts")
        lines.append(f"draft_count: {len(drafts)}")
        lines.append("")
        if input_data is not None:
            lines.append("## Original Prompt")
            lines.append(str(input_data).strip())
            lines.append("")
        for idx, draft in enumerate(drafts, start=1):
            lines.append(f"## Draft {idx}")
            draft_text = "" if draft is None else str(draft).strip()
            lines.append(draft_text)
            lines.append("")
        return "\n".join(lines).strip()

    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        drafts: List[Any] = []
        draft_metadatas: List[Optional[Dict[str, Any]]] = []
        aggregated_tool_calls: List[Dict[str, Any]] = []

        for draft_index in range(1, self.draft_count + 1):
            try:
                self.logger.info(f"[MultiDraft] Starting draft {draft_index}/{self.draft_count}")
            except Exception:
                pass

            run_context = dict(context or {})
            # Ensure each draft call starts fresh (no previous_response_id chaining)
            run_context.pop("previous_response_id", None)

            result = await self.base_agent.execute(input_data, run_context)
            if not result.success:
                error_msg = f"Draft {draft_index} failed: {result.error}"
                self.logger.error(error_msg)
                return AgentResult(success=False, data=None, error=error_msg, metadata=result.metadata)

            drafts.append(result.data)

            meta = result.metadata or {}
            if meta:
                meta_copy = dict(meta)
                meta_copy.pop("previous_response_id", None)
                draft_metadatas.append(meta_copy)
            else:
                draft_metadatas.append(None)

            if result.tool_calls:
                aggregated_tool_calls.extend(result.tool_calls)

        combined = self.aggregate_fn(input_data, drafts)

        metadata: Dict[str, Any] = {
            "agent_name": self.name,
            "base_agent": getattr(self.base_agent, "name", type(self.base_agent).__name__),
            "draft_count": len(drafts),
            "drafts": drafts,
        }
        if any(draft_metadatas):
            metadata["draft_metadatas"] = draft_metadatas

        combined_payload = {
            "draft_count": len(drafts),
            "drafts": drafts,
            "combined": combined,
        }

        return AgentResult(
            success=True,
            data=combined_payload,
            metadata=metadata,
            tool_calls=aggregated_tool_calls or None,
        )
