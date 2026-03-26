# Direct Pair Sign-Flip Diagnostics

Source panel: `data/panels/RTD_DIRECT_PAIR_PANEL_202512240005_202603242300.parquet`

This memo summarizes the diagnostics in `regressions/direct_pair_sign_flip_diagnostics.html` for the sign flip on `equip_cong_any_1` between the direct-pair non-elasticity and elasticity-style regressions.

## Bottom Line

- In the baseline direct-pair regression, the island-1 equipment congestion coefficient is positive in levels but negative in the elasticity-style specification.
- That positive levels coefficient weakens substantially after winsorizing the top 1% of gaps.
- Once extreme scarcity intervals are removed, the levels coefficient also turns negative.
- The most likely story is that the positive levels estimate is being pulled upward by a relatively small set of extreme price-gap episodes.

## Coefficient Path For `equip_cong_any_1`

| Model | Coef | SE | p-value | N | R-squared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline level | 172.7624 | 110.8047 | 0.1190 | 52,348 | 0.3535 |
| Baseline elasticity-style | -0.1003 | 0.0319 | 0.0017 | 52,348 | 0.8151 |
| Winsorized level | 64.6647 | 70.7799 | 0.3609 | 52,348 | 0.4311 |
| Winsorized elasticity-style | -0.1024 | 0.0317 | 0.0012 | 52,348 | 0.8154 |
| Restricted-sample level | -96.7455 | 54.7384 | 0.0772 | 50,725 | 0.4336 |

Details:

- Winsorization caps `dep_abs_price_gap` at the 99th percentile: `22,280.1501`.
- The scarcity restriction removes `1,623` rows where either island price is `>= 30,000` or `<= -9,000`.

Interpretation:

- The levels coefficient falls from `+172.8` to `+64.7` after trimming the top 1% tail.
- Removing scarcity intervals pushes the levels coefficient to `-96.7`.
- The elasticity-style coefficient stays stably negative before and after winsorization.

## Gap Distribution During Island-1 Congestion Episodes

| Island 1 congested | Rows | Share | Mean gap | Median gap | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 48,880 | 93.38% | 1,510.4620 | 57.7037 | 3,657.6532 | 10,411.5802 | 21,940.3065 | 123,229.2055 |
| 1 | 3,468 | 6.62% | 1,618.2915 | 69.6136 | 4,259.7777 | 9,078.9225 | 23,418.2637 | 86,939.4937 |

Takeaway:

- Congestion on island 1 is associated with somewhat larger mean and median gaps.
- But the unconditional difference is modest relative to the overall tail thickness of the dependent variable.

## Mean And Median Gap By Congestion Indicator

| Indicator | State | Rows | Mean gap | Median gap |
| --- | ---: | ---: | ---: | ---: |
| `link_congested_any` | 0 | 38,349 | 119.6959 | 16.4241 |
| `link_congested_any` | 1 | 13,999 | 5,347.0534 | 2,560.0989 |
| `equip_cong_any_1` | 0 | 48,880 | 1,510.4620 | 57.7037 |
| `equip_cong_any_1` | 1 | 3,468 | 1,618.2915 | 69.6136 |
| `equip_cong_any_2` | 0 | 49,937 | 1,401.4295 | 54.1248 |
| `equip_cong_any_2` | 1 | 2,411 | 3,923.8630 | 1,022.8629 |

Takeaway:

- Link congestion is by far the strongest simple separator of large pair gaps.
- Island-2 equipment congestion has a much larger raw association with gaps than island-1 equipment congestion.
- That helps explain why the island-1 coefficient can be sensitive once controls and tails are handled differently.

## Correlation And VIF

Selected pairwise correlations:

- `corr(link_congested_any, equip_cong_any_1) = -0.0884`
- `corr(link_congested_any, equip_cong_any_2) = 0.2502`
- `corr(equip_cong_any_1, equip_cong_any_2) = -0.0567`
- `corr(losses_total, generation_total) = 0.8557`
- `corr(mkt_import_total, mkt_export_total) = 1.0000`

Full VIF:

| Variable | VIF |
| --- | ---: |
| `link_congested_any` | 1.5131 |
| `equip_cong_any_1` | 1.1264 |
| `equip_cong_any_2` | 1.1664 |
| `losses_total` | 89.3679 |
| `generation_total` | 91.0500 |
| `mkt_import_total` | 21,114,021,291.1007 |
| `mkt_export_total` | 21,114,021,291.1007 |

Reduced VIF dropping `mkt_export_total`:

| Variable | VIF |
| --- | ---: |
| `link_congested_any` | 1.5131 |
| `equip_cong_any_1` | 1.1263 |
| `equip_cong_any_2` | 1.1664 |
| `losses_total` | 89.3679 |
| `generation_total` | 91.0498 |
| `mkt_import_total` | 4.7900 |

Takeaway:

- The congestion indicators themselves do not have alarming VIFs.
- The RTDREG totals do have severe multicollinearity.
- `mkt_import_total` and `mkt_export_total` are effectively identical in this panel, which makes the full-system totals block especially unstable.

## Practical Read

- The sign flip does not look like a pure congestion-indicator collinearity problem.
- It looks much more like a tail and scarcity problem interacting with a highly collinear totals block.
- If the goal is a stable interpretive coefficient on island-1 equipment congestion, the restricted-sample level result and the elasticity-style result point in the same negative direction.
