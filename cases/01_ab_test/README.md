# Case 1 — A/B Test Readout

**Question.** A product experiment nudges users toward posting more often. Does it work, and should it ship?

**Naive answer.** Compare raw posting rates between arms: control 35.4%, treated 41.4%, a **+5.97pp** lift (p < 0.001). Ship it.

**Why that's incomplete.** The naive number isn't wrong here — it holds up under a covariate-adjusted check (+5.98pp, nearly identical) and the test was well-powered (10,082/arm vs. 1,030 required) with no sample ratio mismatch (chi-square p = 0.246). The gap is what the naive read *didn't look at*: guardrail metrics. Among posters, average post length drops ~10% (140 → 126 words, p < 1e-70) and the content-moderation flag rate rises ~1.6pp (3.4% → 5.0%, p < 0.001) in the treatment arm. The nudge produces more posts, but more of them are shorter and more of them get flagged.

**Method.**
1. SRM check — chi-square goodness of fit on arm sizes vs. the intended 50/50 split (`src/causal_toolkit/ab.py::srm_check`).
2. Power check — required vs. actual sample size for an 80%-power test on the planted effect (`power_check`).
3. Primary effect — naive two-proportion comparison and a covariate-adjusted logistic-regression average marginal effect, both with 95% CIs (`naive_effect`, `adjusted_effect`).
4. Guardrails — two-sample tests (continuous and binary) on post length and flag rate, each with a CI and a pass/regressed flag (`guardrail_check_continuous`, `guardrail_check_binary`).
5. Independent cross-check of the topline and guardrail metrics in DuckDB (`sql/01_case1_metrics.sql`).

Data is seeded synthetic data (`causal_toolkit.ab.generate_case1_data`, seed `20260101`) mirroring the schema and rough effect size of the C22/HW1 course case (an online platform's posting-activity A/B test), with a planted true effect — the source has no redistribution license. See `../../.ai/PROJECT_SPEC.md`.

**Result + CI.**

| Check | Result |
|---|---|
| SRM | control 10,082 / treated 9,918, χ²=1.35, p=0.246 — balanced |
| Power | 10,082/arm actual vs. 1,030/arm required for 80% power — adequately powered |
| Primary effect (naive) | +5.97pp [4.62, 7.31], p<0.001 |
| Primary effect (adjusted) | +5.98pp [4.65, 7.31], p<0.001 |
| Guardrail: post length | −14.4 words / −10.2% [−15.9, −12.8], p<0.001 — regressed |
| Guardrail: flag rate | +1.6pp [0.7, 2.5], p<0.001 — regressed |

Figures: `../../reports/figures/case1_effect_ci.png`, `../../reports/figures/case1_guardrails.png`.

**Recommendation.** Do not roll out as-is. Pilot to a smaller cohort with quality monitoring, or redesign the nudge to reward substantive posts (e.g. a length or engagement floor) before re-testing.

**Limitations.** Synthetic data with a planted effect — detection here is not evidence of real-world performance. The p<0.05 / any-direction-of-regression guardrail rule is illustrative, not calibrated to a specific business cost function. No downstream retention or revenue outcome is modeled, only the immediate experiment window.

See `MEMO.md` for the one-page decision memo, and `case1_ab_test.ipynb` for the full executed analysis.
