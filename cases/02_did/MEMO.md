# Case 2 memo — paid-search incrementality

Question: should the sponsored-search program be scaled based on the observed paid-click ROI?

Naive answer: paid clicks appear to return 320.0% revenue per dollar of spend because every paid click is counted as incremental.

Why that is wrong: the campaign cannibalizes organic clicks, and the broad comparison pool fails the pre-period parallel-trends screen (`passes=False`).

Method: use a difference-in-differences model on total clicks, restricted to the comparator markets that pass the same pre-trends check (`passes=True`).

Result: the estimated incremental lift is 45.0 clicks per treated market-week (95% CI 45.0 to 45.0); the planted effect is 45.0.

Recommendation: use the DiD readout, not the paid-click topline; the corrected ROI is 80.0%, so scaling should depend on whether that clears the margin hurdle.

Limitations: this synthetic case assumes stable market composition, no spillovers, and a validated comparable-control screen.
