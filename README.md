# rerank-bench

Benchmarking Jev (TypeSafe SystemOne) as a reranker against NVIDIA's Nemotron VL
cross-encoder reranker (via OpenRouter), on real long documents — books, law, math,
physics, encyclopedic text — scored by gold IR metrics and blind, swap-corrected
LLM-as-judge pairwise comparison.

Work in progress. Methodology: `docs/specs/2026-09-26-rerank-bench-design.md`.
Task breakdown: `docs/plans/2026-09-26-rerank-bench-plan.md`.
