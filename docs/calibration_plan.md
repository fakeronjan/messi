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

---

## Update v2 (2026-06-07): calendar window + China dedup

The first cut (200 game-day window, λ=2) shipped, then surfaced a real flaw:
**idle elites collapsed during pre-tournament friendly lulls.** Argentina fell
#7 → #19 with no loss — its competitive pedigree (2024 Copa, 2025 qualifiers)
aged out of the 200-day window, leaving only weak friendlies, and λ=2 partial
pooling then over-shrank it toward the confed mean. No single λ fixed both the
lurch and the collapse (they trade off monotonically), so the window was the
right lever.

- **Window: 200 game-days → 3 CALENDAR years**, with an exponential **1.5yr
  half-life** (decoupled from the window length). Calendar years keep the
  horizon stable across the lumpy international calendar; the half-life carries
  responsiveness while a light tail of pedigree keeps proven teams alive through
  lulls. Tested 5yr but it **re-propped Brazil** (padded blowouts accumulate);
  3yr is the balance. `confed_prior_lambda` lowered **2.0 → 1.0** (the longer
  window does most of the thin-team work, so a gentler prior suffices).
- **China-PR dedup:** the source renamed `China PR → China` ~2026-04; the
  append-only db-union preserved the orphaned old name, duplicating the entire
  team. Fixed with `TEAM_NAME_NORMALIZATION` (applied to fresh + committed rows
  before dedup) + an **orphan-rename WARNING guard** so the next such rename is
  caught immediately instead of silently duplicating for months.
- **Validation on rebuild:** Argentina **#5** (un-collapsed), China single entry,
  **champions 8/10 at #1**, March lurch +1.10 → **+0.24**, GOAT balanced 10
  UEFA / 9 CONMEBOL. Scale shifted up (top ~3.5, peaks ~4.0) → UI bar ±4 → ±5,
  tiers → ~2 / ~2.7 / ~3.5 (Germany 2014 = 3.73).

### Deferred — next iteration: quality-adjusted confed offset

The one residual is **Brazil 1997 at GOAT #1** (a Copa winner over WC-champion
Germany 2014). Probed and traced to **Layer 2**: CONMEBOL's offset (+0.82, above
UEFA) is created **entirely by Argentina & Brazil** — remove their cross-confed
games and CONMEBOL drops to +0.54, *below* UEFA's +0.86; their direct H2H vs
UEFA is +0.97 with Arg/Bra but −0.14 without. The super-team solve reads the two
giants' quality as the whole confederation's level, then partial pooling stamps
it onto every CONMEBOL team. **Cheap fixes don't work** (a per-team weight cap
moves the level *unpredictably* — the zero-sum super-team solve is too
sensitive). The proper fix is a **quality-adjusted offset**: attribute a
cross-confed result to the participating teams' own strength first and only the
*residual* to the confederation — a joint/iterative estimation (circular with
the within-confed ratings), deserving its own design + validation pass. Brazil
1997 #1 is a defensible interim (a 37-8-2 Copa-winning Ronaldo/Romário dynasty).
