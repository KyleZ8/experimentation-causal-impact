# Memo: Posting Nudge Experiment — Ship Decision

**To:** Product Lead
**Re:** Should we roll out the posting nudge to 100% of users?

## Bottom line
**Not yet.** The nudge works — but it's shipping a quality problem alongside the engagement win. Recommend a targeted pilot with quality monitoring, or a redesign, before a full rollout.

## What we found
- Posting rate rose **+6.0 percentage points** (35.4% → 41.4%), a real and statistically robust effect — confirmed two independent ways (raw comparison and a regression-adjusted estimate), on a properly randomized, adequately powered test (10x the required sample size).
- But among people who posted, **average post length fell ~10%** and the **content-moderation flag rate rose ~1.6 points** (3.4% → 5.0%, a 47% relative increase). The nudge is driving more posts, not necessarily better ones — and a meaningful share of the new volume is getting flagged.

## Why this matters
If we ship on the topline number alone, we optimize for volume and quietly accept more low-effort and policy-violating content. That's a cost the primary metric doesn't show and a pattern moderation/trust-and-safety will feel before growth does.

## Recommendation
1. Do not roll out to 100% this cycle.
2. Pilot to a smaller segment with post-quality and flag-rate monitoring in place.
3. In parallel, test a variant that rewards substantive posts (e.g., a minimum length or engagement threshold before the nudge credits it), then re-run this readout.

## Caveats
Findings are from a controlled analysis on this experiment's window only — no read yet on whether the engagement lift or the quality drop persists past the first few sessions. Full technical detail: `README.md` and `case1_ab_test.ipynb` in this folder.
