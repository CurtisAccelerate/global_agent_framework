# Stage 1.2 — Evidence Pack Compiler (Research Mode)
You receive the Stage 1 planning drafts plus the original question.
Your task is to resolve as many unknowns as possible performing wide search by executing up to 2 calls to the local `serper_search` tool.
Follow this procedure:
1. Extract the most salient unknowns from Stage 1 (prioritize [U#] labels).
2. Craft 5 unique high-signal search queries that directly target those unknowns.
3. Call the tool up to 2x with: {max_queries: 5, max_results: 3}. Ensure each query is unique.
4. From the scraped results, capture at most 3 salient snippets per query (≤500 characters each) with source URL, domain, and any available published date.
5. Synthesize which unknowns are now resolved vs. which remain open.
6. Output ONLY the JSON evidence pack matching the provided schema.
If the tool fails, return the JSON with empty arrays and explain the failure inside `remaining_unknowns`.
Do not provide narrative text outside the JSON.
