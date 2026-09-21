# Case 3 memo — directed-search causal impact

Question: does directed search increase session sales, or are high-intent customers simply more likely to use it?

Naive answer: sessions using directed search show $15.09 more sales than other sessions.

Why that is wrong: high-intent and returning customers self-select into directed search, creating pre-treatment imbalance.

Method: estimate propensity scores from pre-session covariates, nearest-neighbor match treated sessions to comparable controls, and verify balance.

Result: the matched ATT is $7.91 per treated session (95% CI $7.81 to $8.01); the planted effect is $8.00.

Recommendation: credit directed search with the matched effect, not the naive gap; the mean absolute SMD falls from 0.458 to 0.019 across 2,204 matched pairs.

Limitations: PSM only adjusts observed covariates, depends on overlap, and should be paired with product knowledge about unobserved search intent.
