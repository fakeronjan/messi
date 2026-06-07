# MESSI Confederation Calibration

_Shipped to `messi.py` on branch `confed-calibration`, 2026-06-07._

## The problem

MESSI's **within-confederation** ordering was always solid, but the
**between-confederation level** was unstable. The match graph is six dense
confederation clusters joined by thin bridges — and those bridges are mostly
friendlies (down-weighted to 0.25), because competitive games are almost all
intra-confederation. The World Cup (the only heavily-weighted cross-confed
competition) falls outside the 200-game-day window for most of each 4-year
cycle, so mid-cycle the entire cross-confed level was set by a handful of
friendlies.

Symptom: on 2026-03-26, two friendlies (Brazil 1-2 France, Colombia 1-2
Croatia) entered the window and floated the **entire UEFA bloc up ~1.2 rating
points in a day** — Norway went #5 → #2 without playing. Historically, the
model's true #1 was a sub-5-competitive-game phantom on 41% of game-days
(e.g. Israel 1998, ahead of champion France).

## The fix — a confed ↔ team iterative calibration

Two diseases, two tools, composed as one iterative solve:

- **Layer 2 — slow confed offset.** Treat each confederation as a single
  "super-team" and solve the cross-confed games over an 8-year, recency-weighted
  (3yr half-life) window. Produces one calibrated level per confederation.
  Pure function of games — no cache dependency. (`_confed_offset`)

- **Layer 1 — partial pooling.** In the global WLS, add one prior row per team
  pulling its rating toward *its confederation's level* (Layer 2), with weight
  `confed_prior_lambda`. Thin-evidence teams get dragged to "average team in
  their confederation" (kills the Israel/Australia phantoms at the root);
  well-observed teams override the prior and keep their earned rating.
  (`_solve_wls`, prior rows)

- **Layer 3 — anchored re-estimation.** After solving, re-estimate each
  confederation's level from the solved bloc means, but **blended back toward
  the slow Layer-2 offset** (`recenter_anchor_alpha`). Feed that as the prior
  for the next pass. Iterate a fixed `calibration_iters` (K=3) times.

```
c = slow confed offset                                   (Layer 2)
repeat K times:
    ranked = solve teams, each shrunk toward c           (Layer 1)
    c = alpha*offset + (1-alpha)*bloc_means(ranked)       (Layer 3, demeaned)
final = ranked
```

The anchoring is the crucial bit: a naive confed→team→confed loop would
re-converge onto the reactive (lurchy) window level, because the window's games
genuinely say "UEFA is high this week." Anchoring the re-estimate to the slow
offset keeps the loop near the slow level. Within-confed reactivity is fully
preserved (a team can still rise on merit); only the bloc *level* is stabilized.

## Parameters (in `messi.py`)

| Param | Value | Role |
|---|---|---|
| `confed_prior_lambda` | 2.0 | partial-pooling strength (thin-team shrinkage) |
| `confed_offset_years` | 8 | horizon for the slow cross-confed level |
| `confed_offset_halflife_d` | 1095 | recency half-life within that horizon |
| `recenter_anchor_alpha` | 0.6 | weight on slow offset vs solved bloc level in re-estimation |
| `calibration_iters` | 3 | fixed K confed↔team passes per snapshot |

Unchanged & locked: `window_game_days=200`, `margin_cap=8`, `home_field_adv=0.5`,
`min_competitive_games=5` (now a clean editorial floor, no longer the primary
anti-phantom defense — ridge handles that; the floor still catches the
1-game tail like Guyana-1998).

## Validation (offline prototype + engine-function test, 2026-06-07)

- World Cup champion backtest: **5/7 at #1, 7/7 top-3** (was robust before; held).
- Israel-1998 phantom: #1 → **#8** (France correctly #1). Australia-2006: → **#13**.
- March-2026 UEFA bloc lurch: **+1.10 → +0.10**. Norway: stable **#1 / #2 / #1**
  across Feb/Mar/Jun — bold but earned and smooth, not a spike.
- GOAT list: canonical greats survive (Germany '14, Brazil '02, Spain '12,
  Argentina '16/'22), top-20 spread slightly *wider* than production (no
  compression-into-mush — the failure mode that sank the 400-day window test).

## Caveats / downstream

- **Cache:** the methodology change is invisible to the cache-validity guard
  (it checks id↔date mapping, not formula). `messi_ratings.csv.gz` and
  `messi_ratings_final.csv.gz` **must be deleted** before rebuilding, or it
  serves stale old-formula ratings.
- **Rebuild:** full, from-scratch, **foreground/serial** (never parallel — see
  the rebuild-hang lesson). Per-snapshot cost is ~K× the old solve; the slow
  offset is recomputed once per calendar month to keep it tractable.
- **Rating scale compresses** (top team ~2.6 vs ~4.7 before). Audit the header
  model-note benchmark tiers and any chart/bar ranges in `generate_data.py` /
  UI; the JSON column schema is unchanged, only the values move.
- **Rollout:** this closes the original Alpha reason (Israel-98) at the root
  plus the deeper lurch. Combined with live WC 2026 cross-confed data, it's the
  natural Alpha → Beta trigger — gate promotion on WC behaving sanely under the
  new calibration, not just the backtest.
