**Stage 2: Guided Uncertainty Resolution & Self-Confidence Loop**
- **Objective:** Systematically resolve the most salient unknowns identified in Stage 1 and 1.2, with strong preference for using function tools where APPLICABLE. Tools provide market odds which should prioritized over free-text search.
- **Tool preference rules:**
- For **sports outcomes** (football, boxing, etc.): MUST TRY the 'odds_find' function tool (single call returns de-vigged consensus probabilities and best prices). Use web search only to supplement or cross-check.
- For **financial/crypto markets**: MUST TRY Deribit function tools ('deribit_weekly_snapshot', 'deribit_weekly_ladder', 'deribit_weekly_inputs') to extract options/IV/strike ladders. Use web search only to add macro context or if tool returns not relevant data.
- For **event-driven prediction markets**: DO PREFER 'polymarket_gamma_get_odds' to pull Polymarket event odds, rules, and liquidity. Use web search to cross-validate news or market context.
- Web search is ENCOURAGED as a complement (e.g., event logistics, injury reports, macro calendar), for complex open-ended queries where tools dont support, or where data not available or not relevant but it should not replace function tool outputs when a relevant function tool exists.
- Web search is REQUIRED for complex/non-relevant queries that are not supported by the function tools.
- Web search is REQUIRED if tools fail or don't return relevant data.
- You *do* have a web_search tool and you should *always* use to complement the function tool outputs.
### Step 2.0 — Review Draft Contexts
You will receive Stage 1 output that may contain multiple drafts labelled `Draft 1`, `Draft 2`, etc. Review all drafts and synthesize a unified plan that captures the best insights, unknowns, and tool strategies across them.
Highlight any discrepancies between drafts and resolve them explicitly before moving forward.
### Step 2.1 — Plan for Each Unknown
For every identified unknown (**[U#]**) across all drafts, briefly describe how you will address it—prioritize using search to obtain evidence or examples. Cite all sources. Deprioritize constructing composite lines; instead, focus on isolating and analyzing the most salient qualities or facts relevant to the uncertainty.
### Step 2.2 — Execute Resolution
Carry out the planned approach for each unknown, surfacing 3–5 relevant and diverse examples or pieces of evidence per unknown as required. Be explicit about how each piece of evidence impacts your understanding. Continuously self-assess: For each line of reasoning or finding, state your confidence as a percentage.
### Step 2.3 — Loop and Update Confidence
If confidence in your current understanding or answer is below 95%, clearly identify the remaining or new uncertainty. Loop back to Step 2.1 for unresolved points. Repeat this process until at least 95% confidence is achieved. Only then proceed to final synthesis.
**Note:** Throughout this stage, emphasize salient points and evidence; discourage complex composites focusing on salience, entailment. You must state/show your entailments and citations in the output for future stage prompts to use.            

You must return your best prediction in the response/text even if you don't meet 95% confidence.

Pay particular attention to letter to price mapping in the original question if provided: DO NOT ATTEMPT TO REMAP, NORMALIZE, or CHANGE IT. Your final answer must adhere to the format specified in the original question.

Caution: For final answer, absolutely do not re-map or normalize the answer choices under any circumstances. The answer choices are immutable, ground-truth and must be emitted correctly. Do not assume they are normalized.

Do strongly prefer adjuncting search with custom tools polymarket, deribit, odds_fine where applicable.
