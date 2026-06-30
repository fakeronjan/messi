"""Biggest knockout upsets across all WC editions: for each played KO game,
the WINNER's pre-game win probability (Poisson goals model + shootout logit,
copied from generate_data.py / retro_wc.py). Lowest probability = biggest upset.
Dry run prints the leaderboard; --write emits docs/data/wc_upsets.json."""
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from bisect import bisect_left

REPO = os.path.dirname(os.path.abspath(__file__))   # run from the repo root

R = pd.read_csv(f'{REPO}/messi_ratings_final.csv.gz', usecols=['date', 'country', 'rating', 'rank'])
R['date'] = pd.to_datetime(R['date']).dt.date
G = pd.read_csv(f'{REPO}/all_soccer_games.csv', low_memory=False)
G['date'] = pd.to_datetime(G['date']); G['d'] = G['date'].dt.date; G['yr'] = G['date'].dt.year
NEUTRAL = G['neutral'] if 'neutral' in G.columns else pd.Series(False, index=G.index)
FLAG = {t['name']: t.get('flag', '') for t in json.load(open(f'{REPO}/docs/data/teams_index.json'))}

SNAP = {}
for team, sub in R.sort_values('date').groupby('country'):
    SNAP[team] = (sub['date'].tolist(), sub['rating'].tolist(), sub['rank'].tolist())


def _pre(team, d):
    """(rating, rank) strictly before date d (the going-in value), or (None, None)."""
    s = SNAP.get(team)
    if not s:
        return (None, None)
    i = bisect_left(s[0], d)
    if i <= 0:
        return (None, None)
    return (s[1][i - 1], None if pd.isna(s[2][i - 1]) else int(s[2][i - 1]))


def rating_pre(team, d):
    return _pre(team, d)[0]


# ── Poisson fit (copied) ────────────────────────────────────────────────────
_rows = []
for _i, _g in G.iterrows():
    if pd.isna(_g['home_score']) or pd.isna(_g['away_score']) or pd.isna(_g['date']):
        continue
    _rh, _ra = rating_pre(_g['home_team'], _g['d']), rating_pre(_g['away_team'], _g['d'])
    if _rh is None or _ra is None:
        continue
    _rows.append((_rh - _ra, 0.0 if bool(NEUTRAL.iloc[_i]) else 1.0, int(_g['home_score']), int(_g['away_score'])))
_gm = np.array(_rows, float)
_X = np.vstack([np.column_stack([np.ones(len(_gm)), _gm[:, 0], _gm[:, 1]]),
                np.column_stack([np.ones(len(_gm)), -_gm[:, 0], -_gm[:, 1]])])
_yv = np.concatenate([_gm[:, 2], _gm[:, 3]])
_b = np.zeros(3)
for _ in range(25):
    _lam = np.exp(_X @ _b)
    _b += np.linalg.solve((_X * _lam[:, None]).T @ _X, _X.T @ (_yv - _lam))
MU, A, _B = _b

_FACT = [math.factorial(k) for k in range(20)]


def win_prob(rw, rl):
    """P(winner beats loser) in one neutral KO game: Poisson goals + shootout logit."""
    d = rw - rl
    lw, ll = math.exp(MU + A * d), math.exp(MU - A * d)
    K = 16
    pw = [math.exp(-lw) * lw**k / _FACT[k] for k in range(K)]
    pl = [math.exp(-ll) * ll**k / _FACT[k] for k in range(K)]
    p_more = sum(pw[i] * pl[j] for i in range(K) for j in range(K) if i > j)
    p_draw = sum(pw[i] * pl[i] for i in range(K))
    return p_more + p_draw / (1.0 + math.exp(-0.25 * d))


def ko_winner(r):
    if r['home_score'] > r['away_score']:
        return r['home_team']
    if r['away_score'] > r['home_score']:
        return r['away_team']
    so = r.get('shootout_winner')
    return so if isinstance(so, str) and so.strip() else None


STAGE = {32: 'R32', 16: 'R16', 8: 'QF', 4: 'SF', 2: 'Final'}
upsets = []
for yr in range(1986, 2027):
    e = G[(G['tournament'] == 'FIFA World Cup') & (G['yr'] == yr)].sort_values('date').reset_index(drop=True)
    if not len(e):
        continue
    nt = pd.concat([e['home_team'], e['away_team']]).nunique(); ng = nt // 4
    ko = e.iloc[ng * 6:]
    cnt, elim, r1games = {}, set(), 0
    games = []
    for _, r in ko.iterrows():
        a, b = r['home_team'], r['away_team']
        if a in elim and b in elim:                          # 3rd-place playoff
            continue
        rd = max(cnt.get(a, 0), cnt.get(b, 0)) + 1
        if rd == 1:
            r1games += 1
        if pd.isna(r['home_score']):                          # unplayed (in-progress 2026)
            continue
        w = ko_winner(r)
        l = b if w == a else a
        games.append((rd, w, l, r))
        cnt[a] = cnt.get(a, 0) + 1; cnt[b] = cnt.get(b, 0) + 1; elim.add(l)
    ko_size = 2 * r1games if r1games else 16
    for rd, w, l, r in games:
        (rw, rkw), (rl, rkl) = _pre(w, r['d']), _pre(l, r['d'])
        if rw is None or rl is None:
            continue
        wp = win_prob(rw, rl)
        gf, ga = (int(r['home_score']), int(r['away_score'])) if r['home_team'] == w else (int(r['away_score']), int(r['home_score']))
        so = r['shootout_winner'] if isinstance(r.get('shootout_winner'), str) and r['shootout_winner'].strip() else None
        upsets.append({
            'edition': yr, 'round': STAGE.get(ko_size // (2 ** (rd - 1)), f'R{ko_size//2**(rd-1)}'),
            'winner': w, 'winner_flag': FLAG.get(w, ''), 'winner_rating': round(rw, 2), 'winner_rank': rkw,
            'loser': l, 'loser_flag': FLAG.get(l, ''), 'loser_rating': round(rl, 2), 'loser_rank': rkl,
            'gf': gf, 'ga': ga, 'pens': bool(so), 'date': str(r['d']), 'win_prob': round(wp, 4),
        })

upsets.sort(key=lambda u: u['win_prob'])
print(f"{len(upsets)} knockout games scored. Top 18 upsets (winner's pre-game odds):")
for u in upsets[:18]:
    p = 'p' if u['pens'] else ''
    print(f"  {u['win_prob']*100:4.1f}%  {u['edition']} {u['round']:5s}  {u['winner']} {u['gf']}-{u['ga']}{p} {u['loser']}")

if '--write' in sys.argv:
    json.dump({'upsets': upsets}, open(f'{REPO}/docs/data/wc_upsets.json', 'w'), separators=(',', ':'))
    print(f"WROTE {len(upsets)} to wc_upsets.json")
