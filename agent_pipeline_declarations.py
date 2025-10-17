"""
Agent pipeline declarations: create and configure example pipelines used by the API and tests.
"""
from pipeline import Pipeline
from agent import Agent, AgentResult, GPTAgent, ReasoningConfig, ToolDefinition, AgentsSDKAgent, MultiDraftAgent
from config import Config
from typing import Optional, Dict
from pathlib import Path

try:
    from tools.polymarket_gamma import polymarket_gamma_get_odds
except Exception:
    polymarket_gamma_get_odds = None

try:
    from tools.deribit import (
        deribit_weekly_snapshot,
        deribit_weekly_ladder,
        deribit_weekly_inputs,
    )
except Exception:
    deribit_weekly_snapshot = None
    deribit_weekly_ladder = None
    deribit_weekly_inputs = None

try:
    from tools.odds_api import odds_find, odds_get
except Exception:
    odds_find = None
    odds_get = None

try:
    from tools.serper import serper_search
except Exception:
    serper_search = None

try:
    from tools.serp_dev_stub import serp_dev_search_stub
except Exception:
    serp_dev_search_stub = None

PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_pipelines"
USER_QUERY_DIR = Path(__file__).resolve().parent / "queries"


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt template: {path}") from exc


def load_user_query(name: str) -> str:
    path = USER_QUERY_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing user query template: {path}") from exc


def create_prediction_pipeline(user_query_template: Optional[str] = None) -> Pipeline:
    """Create the declarative prediction pipeline using built-in tools and defaults."""
    # Stage 1 — build tool-aware context (Stage 1 first in file and pipeline)

    if user_query_template:
        input_payload: Dict[str, str] = {
            "user_query": load_user_query(user_query_template)
        }
    else:
        input_payload = {}

    context_generator_base = AgentsSDKAgent(
        name="context_generator",
        system_prompt=load_prompt("futurerx/context_generator.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="minimal",
            max_output_tokens=8192,
            tool_choice="none",
            verbosity="medium",
            text_format_type="text",
            timeout_seconds=300,
        ),
    )

    context_generator = MultiDraftAgent(
        base_agent=context_generator_base,
        draft_count=2,
        name="context_generator"
    )

    evidence_builder = AgentsSDKAgent(
        name="evidence_builder",
        system_prompt=load_prompt("futurerx/evidence_builder.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="medium",
            max_output_tokens=4096,
            tools=[],
            tool_choice="required",
            max_tool_calls=2,
            parallel_tool_calls=False,
            timeout_seconds=400
        ),
    )

    serp_tool_impl = serper_search if not getattr(Config, "SERPER_USE_STUB", False) else serp_dev_search_stub
    if serp_tool_impl is not None:
        evidence_builder.add_tool(ToolDefinition(
            name="serper_search",
            description=(
                "Run up to 5 serper.dev searches in a single call and return top 2 organic results with scraped content. "
                "Always returns structured evidence suitable for building research packets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "description": "Unique list of search queries targeting Stage 1 unknowns.",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    },
                    "max_queries": {"type": ["integer", "null"], "default": 5},
                    "max_results": {"type": ["integer", "null"], "default": 2}
                },
                "required": ["queries"],
                "additionalProperties": False
            },
            function=serp_tool_impl,
        ))

    # Stage 2 — Deep Researcher
    deep_researcher = AgentsSDKAgent(
        name="deep_researcher",
        system_prompt=load_prompt("futurerx/deep_researcher.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="high",
            max_output_tokens=32768,
            tools=[
                {"type": "web_search"},
            ],
            tool_choice="auto",
            verbosity="high",
            parallel_tool_calls=True,
            max_tool_calls=7,
            extra_request_kwargs={"include": ["web_search_call.action.sources"]},
            timeout_seconds=600,
        ),
    )

    # Register local Polymarket Gamma function tool if available
    if polymarket_gamma_get_odds is not None:
        deep_researcher.add_tool(ToolDefinition(
            name="polymarket_gamma_get_odds",
            description=(
                "Fetch Polymarket market odds via the public Gamma API. Provide at least one of: "
                "query, market_id, or event_id. Defaults: active=true, limit=10. Returns simplified markets with outcome odds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": ["string", "null"], "description": "Free-text search (e.g., 'US election 2024')"},
                    "market_id": {"type": ["string", "null"], "description": "Exact Polymarket market id"},
                    "event_id": {"type": ["string", "null"], "description": "Polymarket event id"},
                    "active": {"type": ["boolean", "null"], "description": "Only active markets; default true"},
                    "limit": {"type": ["integer", "null"], "description": "Max markets to return; default 10 (<=200)"},
                },
                "required": [],
                "additionalProperties": False,
            },
            function=polymarket_gamma_get_odds,
        ))

    # Register Deribit tools if available
    if deribit_weekly_snapshot is not None:
        deep_researcher.add_tool(ToolDefinition(
            name="deribit_weekly_snapshot",
            description=(
                "Return latest daily close for PERPETUAL, nearest weekly option expiry to target_date, and ATM IV/prices."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "enum": ["BTC", "ETH", "SOL"]},
                    "target_date": {"type": "string", "description": "YYYY-MM-DD target resolution date"},
                    "lookback_days": {"type": ["integer", "null"], "default": 7},
                },
                "required": ["currency", "target_date"],
                "additionalProperties": False,
            },
            function=deribit_weekly_snapshot,
        ))
    if deribit_weekly_ladder is not None:
        deep_researcher.add_tool(ToolDefinition(
            name="deribit_weekly_ladder",
            description=(
                "Return ±N-strike ladder around center_strike for a given expiry, including mark_iv and mark_price."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "enum": ["BTC", "ETH", "SOL"]},
                    "expiry_date": {"type": "string", "description": "YYYY-MM-DD expiry date"},
                    "center_strike": {"type": ["number", "integer"]},
                    "width": {"type": ["integer", "null"], "default": 2},
                    "both_sides": {"type": ["boolean", "null"], "default": True},
                },
                "required": ["currency", "expiry_date", "center_strike"],
                "additionalProperties": False,
            },
            function=deribit_weekly_ladder,
        ))
    if deribit_weekly_inputs is not None:
        deep_researcher.add_tool(ToolDefinition(
            name="deribit_weekly_inputs",
            description=(
                "Convenience: snapshot plus a ±1-strike ladder around ATM in one call."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "enum": ["BTC", "ETH", "SOL"]},
                    "target_date": {"type": "string", "description": "YYYY-MM-DD target resolution date"},
                    "lookback_days": {"type": ["integer", "null"], "default": 7},
                },
                "required": ["currency", "target_date"],
                "additionalProperties": False,
            },
            function=deribit_weekly_inputs,
        ))

    # Register Odds API tools if available
    if odds_find is not None:
        deep_researcher.add_tool(ToolDefinition(
            name="odds_find",
            description=(
                "Get sports odds via free-text query. Returns a simplified list of matches and outrights with de‑vigged consensus probabilities and best_price per outcome."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Query like 'Juventus vs Inter', 'Packers', 'Super Bowl 2026 winner'"}
                },
                "required": ["q"],
                "additionalProperties": False,
            },
            function=odds_find,
        ))

    # Stage 2 primarily handles deep reasoning; serp_dev evidence is prepared by stage 1.2 and reused if needed.


    # Stage 3 — use Agents SDK to simplify final formatting (no tools)
    formatter = AgentsSDKAgent(
        name="formatter",
        system_prompt=load_prompt("futurerx/formatter.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="medium",
            max_output_tokens=1024,
            tool_choice="none",
            verbosity="low",
            timeout_seconds=300
        )
    )

    pipeline = Pipeline(
        name="futuerex",
        description="Legacy prediction pipeline: Context → Deep Research (95% confidence) → Formatting & Compliance",
    )
    pipeline.add_stages([context_generator, evidence_builder, deep_researcher, formatter])
    if input_payload:
        pipeline.set_context(input_payload)
    return pipeline




def create_prediction_pipeline_stub() -> Pipeline:
    """Create a pipeline identical to prediction_pipeline but using stub data everywhere."""

    class StubContextAgent(Agent):
        def __init__(self):
            super().__init__("context_generator")

        async def execute(self, input_data, context=None) -> AgentResult:
            drafts = [
                "Stub Draft 1: Enumerate unknowns and plan queries.",
                "Stub Draft 2: Alternative framing of the forecasting task.",
            ]
            data = {
                "drafts": drafts,
                "combined": "Combined stub research framework with unknowns [U1]-[U7].",
                "draft_count": len(drafts),
            }
            return AgentResult(success=True, data=data, metadata={"agent_name": self.name})

    class StubEvidenceAgent(Agent):
        def __init__(self):
            super().__init__("evidence_builder")

        async def execute(self, input_data, context=None) -> AgentResult:
            queries = [
                "stub market outlook",
                "stub regulatory update",
                "stub competitor movement",
                "stub macro trend",
                "stub sentiment shift",
            ]
            payload = await serp_dev_search_stub(queries=queries, max_queries=4, max_results=3)
            evidence = []
            for block in payload.get("results", []):
                for doc in block.get("scraped_documents", [])[:2]:
                    content = doc.get("content")
                    snippet = None
                    if isinstance(content, str) and content.strip():
                        snippet = " ".join(content.split())[:500]
                    if not snippet:
                        snippet = ""
                    evidence.append({
                        "url": doc.get("link"),
                        "domain": doc.get("domain") or "example.com",
                        "date": doc.get("date"),
                        "snippet": snippet,
                    })
            pack = {
                "evidence": evidence,
                "resolved_unknowns": ["[U1] Stub resolved via offline data"],
                "remaining_unknowns": ["[U2] Requires further research"],
            }
            return AgentResult(success=True, data=pack, metadata={"agent_name": self.name})

    class StubResearchAgent(Agent):
        def __init__(self):
            super().__init__("deep_researcher")

        async def execute(self, input_data, context=None) -> AgentResult:
            summary = "Stage 2 stub synthesis referencing offline evidence pack."
            return AgentResult(success=True, data=summary, metadata={"agent_name": self.name})

    class StubFormatterAgent(Agent):
        def __init__(self):
            super().__init__("formatter")

        async def execute(self, input_data, context=None) -> AgentResult:
            boxed = "\\boxed{Stub Result}"
            return AgentResult(success=True, data=boxed, metadata={"agent_name": self.name})

    pipeline = Pipeline(
        name="futuerex_stub",
        description="Stubbed futuerex pipeline without external API calls",
    )
    pipeline.add_stages([
        StubContextAgent(),
        StubEvidenceAgent(),
        StubResearchAgent(),
        StubFormatterAgent(),
    ])
    return pipeline


def create_research_pipeline(user_query_template: Optional[str] = None) -> Pipeline:
    """Create a generalized research pipeline focused on deep evidence gathering and synthesis."""

    pipeline_context: Dict[str, str] = {}
    if user_query_template:
        pipeline_context["user_query"] = load_user_query(user_query_template)

    context_planner_base = AgentsSDKAgent(
        name="context_planner",
        system_prompt=load_prompt("research/context_planner.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="minimal",
            max_output_tokens=8192,
            tool_choice="none",
            verbosity="medium",
            text_format_type="text",
            timeout_seconds=300,
        ),
    )

    context_planner = MultiDraftAgent(
        base_agent=context_planner_base,
        draft_count=2,
        name="context_planner"
    )

    evidence_text_schema = {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "domain": {"type": "string"},
                        "date": {"type": ["string", "null"]},
                        "snippet": {"type": "string", "maxLength": 500}
                    },
                    "required": ["url", "domain", "snippet"],
                    "additionalProperties": False
                }
            },
            "resolved_unknowns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "remaining_unknowns": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["evidence", "resolved_unknowns", "remaining_unknowns"],
        "additionalProperties": False
    }

    research_evidence_builder = AgentsSDKAgent(
        name="research_evidence_builder",
        system_prompt=load_prompt("research/evidence_compiler.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="medium",
            max_output_tokens=4096,
            tools=[],
            tool_choice="required",
            max_tool_calls=2,
            parallel_tool_calls=False,
            timeout_seconds=400
        ),
    )

    serp_tool_impl = serper_search if not getattr(Config, "SERPER_USE_STUB", False) else serp_dev_search_stub
    if serp_tool_impl is not None:
        research_evidence_builder.add_tool(ToolDefinition(
            name="serper_search",
            description=(
                "Run up to 5 serper.dev searches in a single call and return top 2 organic results with scraped content. "
                "Always returns structured evidence suitable for building research packets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "description": "Unique list of search queries targeting Stage 1 unknowns.",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    },
                    "max_queries": {"type": ["integer", "null"], "default": 5},
                    "max_results": {"type": ["integer", "null"], "default": 2}
                },
                "required": ["queries"],
                "additionalProperties": False
            },
            function=serp_tool_impl,
        ))

    deep_researcher = AgentsSDKAgent(
        name="research_deep_researcher",
        system_prompt=load_prompt("research/deep_researcher.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="high",
            max_output_tokens=32768,
            tools=[
                {"type": "web_search"},
            ],
            tool_choice="auto",
            verbosity="high",
            parallel_tool_calls=True,
            max_tool_calls=7,
            extra_request_kwargs={"include": ["web_search_call.action.sources"]},
            timeout_seconds=600,
        ),
    )

    synthesizer = AgentsSDKAgent(
        name="research_synthesizer",
        system_prompt=load_prompt("research/synthesizer.md"),
        model="gpt-5",
        reasoning_config=ReasoningConfig(
            reasoning_effort="medium",
            max_output_tokens=4096,
            tool_choice="none",
            verbosity="medium",
            timeout_seconds=300
        )
    )

    research_pipeline = Pipeline(
        name="research_pipeline",
        description="Generalized research pipeline: Context → Evidence → Deep Research → Synthesis",
    )
    research_pipeline.add_stages([
        context_planner,
        research_evidence_builder,
        deep_researcher,
        synthesizer,
    ])

    if pipeline_context:
        research_pipeline.set_context(pipeline_context)

    return research_pipeline



