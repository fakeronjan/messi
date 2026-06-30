"""Calibration of the historical World Cup knockout odds: does an X% prediction
come true X% of the time? Reads docs/data/wc_odds_history.json. For each completed
edition (16-team knockout, 1990-2022), at every snapshot we take each ALIVE team's
forward-stage probability (reach QF / SF / Final / Champion) and pair it with whether
they actually got there. Bin by predicted probability, compare to realized frequency.

Dry run prints the reliability table; --write emits docs/data/wc_calibration.json
(headline accuracy metric for the site + the bins). Wire into the daily cron so it
re-validates as new editions complete."""
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(f'{REPO}/docs/data/wc_odds_history.json'))
STAGES = ['r16', 'qf', 'sf', 'final', 'champ']   # snapshot order: pre-R16 .. done

pairs, years = [], []
for ed in H['tournaments']:
    snaps = sorted(ed['snapshots'], key=lambda s: -s['games_left'])
    if len(snaps) != 5 or snaps[-1]['games_left'] != 0:
        continue                                  # completed 16-team editions only (skips live 2026)
    years.append(ed['edition'])
    elim = [{t['team']: t['eliminated'] for t in s['teams']} for s in snaps]
    teams = [t['team'] for t in snaps[0]['teams']]
    reached = {tm: [not elim[k].get(tm, True) for k in range(5)] for tm in teams}
    for j, s in enumerate(snaps):
        for t in s['teams']:
            if t['eliminated']:
                continue                          # eliminated -> forward probs are 0, skip
            for k in range(j + 1, 5):
                p = t.get(STAGES[k])
                if p is not None:
                    pairs.append((p, 1 if reached[t['team']][k] else 0))

# Reliability by decile + Expected Calibration Error (avg gap between what the odds
# said and what happened, weighted by how many predictions fell in each band).
deciles = [[] for _ in range(10)]
for p, o in pairs:
    deciles[min(9, int(p * 10))].append((p, o))
bins, ece = [], 0.0
for i, b in enumerate(deciles):
    if not b:
        continue
    pred = sum(p for p, _ in b) / len(b)
    act = sum(o for _, o in b) / len(b)
    ece += len(b) * abs(pred - act)
    bins.append({'lo': i * 10, 'hi': i * 10 + 10, 'n': len(b),
                 'pred': round(pred * 100, 1), 'actual': round(act * 100, 1)})
ece = ece / len(pairs)
brier = sum((p - o) ** 2 for p, o in pairs) / len(pairs)

out = {'avg_error_pts': round(ece * 100, 1), 'brier': round(brier, 4),
       'n_predictions': len(pairs), 'first_year': min(years), 'last_year': max(years),
       'n_editions': len(years), 'bins': bins}

print(f"Calibration over {out['n_editions']} World Cups ({out['first_year']}-{out['last_year']}), "
      f"{out['n_predictions']} predictions:")
print(f"{'band':>9} {'n':>5} {'odds said':>10} {'happened':>9}")
for b in bins:
    print(f"  {b['lo']:2d}-{b['hi']:3d}% {b['n']:>5} {b['pred']:>9.1f}% {b['actual']:>8.1f}%")
print(f"avg error = {out['avg_error_pts']} pts  |  Brier = {out['brier']}")

if '--write' in sys.argv:
    json.dump(out, open(f'{REPO}/docs/data/wc_calibration.json', 'w'), separators=(',', ':'))
    print("WROTE docs/data/wc_calibration.json")
