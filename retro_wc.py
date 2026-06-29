"""One-time retro generator: per-round WC odds snapshots for past editions
(1986-2022), current ratings as of each round, reusing the EXACT Poisson +
knockout-sim model from generate_data.py (copied). Writes the snapshots into
docs/data/wc_odds_history.json, namespaced by (tournament, edition); leaves the
existing 2026 entry untouched. Knockout-only: qualifiers + bracket are ground
truth from results, so no group-stage sim is needed."""
import json
import os
import sys
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right

REPO = os.path.dirname(os.path.abspath(__file__))   # run from the repo root
WRITE = '--write' in sys.argv

R = pd.read_csv(f'{REPO}/messi_ratings_final.csv.gz', usecols=['date', 'country', 'rating', 'rank'])
R['date'] = pd.to_datetime(R['date']).dt.date
G = pd.read_csv(f'{REPO}/all_soccer_games.csv', low_memory=False)
G['date'] = pd.to_datetime(G['date']); G['d'] = G['date'].dt.date; G['yr'] = G['date'].dt.year
NEUTRAL = G['neutral'] if 'neutral' in G.columns else pd.Series(False, index=G.index)
FLAG = {t['name']: t.get('flag', '') for t in json.load(open(f'{REPO}/docs/data/teams_index.json'))}

SNAP = {}
for team, sub in R.sort_values('date').groupby('country'):
    SNAP[team] = (sub['date'].tolist(), sub['rank'].tolist(), sub['rating'].tolist())


def rating_pre(team, d):
    s = SNAP.get(team)
    if not s:
        return None
    i = bisect_left(s[0], d)
    return s[2][i - 1] if i > 0 else None


def rating_asof(team, d):
    s = SNAP.get(team)
    if not s:
        return None
    i = bisect_right(s[0], d) - 1
    return s[2][i] if i >= 0 else None


def rank_asof(team, d):
    s = SNAP.get(team)
    if not s:
        return None
    i = bisect_right(s[0], d) - 1
    return int(s[1][i]) if i >= 0 else None


# ── Poisson model (copied from generate_data.py) ────────────────────────────
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
_beta = np.zeros(3)
for _ in range(25):
    _lam = np.exp(_X @ _beta)
    _beta += np.linalg.solve((_X * _lam[:, None]).T @ _X, _X.T @ (_yv - _lam))
GM_MU, GM_A, GM_B = _beta
print(f"goals model: mu={GM_MU:.3f} a={GM_A:.3f} b={GM_B:.3f} (fit on {len(_gm):,} matches)")

# ── Vectorized KO sim (copied) ──────────────────────────────────────────────
_RNG = np.random.default_rng(20260101)
STAGE = {32: 'r32', 16: 'r16', 8: 'qf', 4: 'sf', 2: 'final', 1: 'champ'}
KEYS = ('r32', 'r16', 'qf', 'sf', 'final', 'champ')
NSIM, CHUNK = 1_000_000, 200_000
KO_LABELS = {16: ['R16', 'QF', 'SF', 'Final']}   # 16-team knockout (all past editions)


def vec_ko(leaf, ratv, NT, reach, played_pos):
    cur = leaf
    reach[STAGE[cur.shape[1]]] += np.bincount(cur.ravel(), minlength=NT)
    rnd = 0
    while cur.shape[1] > 1:
        a = cur[:, 0::2]; b = cur[:, 1::2]; d = ratv[a] - ratv[b]
        ga = _RNG.poisson(np.exp(GM_MU + GM_A * d)); gb = _RNG.poisson(np.exp(GM_MU - GM_A * d))
        a_wins = np.where(ga == gb, _RNG.random(d.shape) < 1.0 / (1.0 + np.exp(-0.25 * d)), ga > gb)
        cur = np.where(a_wins, a, b)
        for (pr, pj), wcol in played_pos.items():
            if pr == rnd:
                cur[:, pj] = wcol
        reach[STAGE[cur.shape[1]]] += np.bincount(cur.ravel(), minlength=NT)
        rnd += 1


def ko_winner(r):
    if r['home_score'] > r['away_score']:
        return r['home_team']
    if r['away_score'] > r['home_score']:
        return r['away_team']
    so = r.get('shootout_winner')
    return so if isinstance(so, str) and so.strip() else None


def edition(yr):
    e = G[(G['tournament'] == 'FIFA World Cup') & (G['yr'] == yr)].sort_values('date').reset_index(drop=True)
    nt = pd.concat([e['home_team'], e['away_team']]).nunique(); ng = nt // 4
    grp, ko = e.iloc[:ng * 6], e.iloc[ng * 6:]
    cnt, elim, games = {}, set(), []
    for _, r in ko.iterrows():
        a, b = r['home_team'], r['away_team']
        if a in elim and b in elim:
            continue
        rd = max(cnt.get(a, 0), cnt.get(b, 0)) + 1
        w = ko_winner(r)
        so = r['shootout_winner'] if isinstance(r.get('shootout_winner'), str) and r['shootout_winner'].strip() else None
        games.append(dict(a=a, b=b, hg=int(r['home_score']), ag=int(r['away_score']), w=w, so=so, rd=rd, date=r['d']))
        cnt[a] = cnt.get(a, 0) + 1; cnt[b] = cnt.get(b, 0) + 1; elim.add(b if w == a else a)
    nrd = max(g['rd'] for g in games)
    rounds = {}
    for g in games:
        rounds.setdefault(g['rd'], []).append(g)
    tg = {}
    for g in games:
        tg[(g['a'], g['rd'])] = g; tg[(g['b'], g['rd'])] = g
    pwin = {frozenset({g['a'], g['b']}): g['w'] for g in games}
    won = {(g['rd'], g['w']): (g['a'], g['b']) for g in games}

    def expand(team, rd):
        if rd == 0:
            return [team]
        ab = won.get((rd, team))
        return [team] if ab is None else expand(ab[0], rd - 1) + expand(ab[1], rd - 1)
    fin = rounds[nrd][0]
    order = expand(fin['a'], nrd - 1) + expand(fin['b'], nrd - 1)
    rend = {rd: max(g['date'] for g in gs) for rd, gs in rounds.items()}
    return dict(order=order, tg=tg, pwin=pwin, nrd=nrd, champ=fin['w'],
                last_grp=grp['d'].max(), rend=rend, nt=nt)


def simulate(order, pwin, asof, pin_below):
    tix = {t: j for j, t in enumerate(order)}; NT = len(order)
    ratv = np.array([(rating_asof(t, asof) or 0.0) for t in order], float)
    played_pos = {}; det = list(order); r = 0
    while len(det) > 1:
        nd = []
        for j in range(0, len(det), 2):
            k = frozenset({det[j], det[j + 1]})
            if (r + 1) < pin_below and k in pwin:
                w = pwin[k]; played_pos[(r, j // 2)] = tix[w]; nd.append(w)
            else:
                nd.append(None)
        det = nd; r += 1
    leaf0 = np.array([tix[t] for t in order], np.int32)
    reach = {k: np.zeros(NT, np.int64) for k in KEYS}
    done = 0
    while done < NSIM:
        n = min(CHUNK, NSIM - done)
        vec_ko(np.tile(leaf0, (n, 1)), ratv, NT, reach, played_pos)
        done += n
    return {t: {k: int(reach[k][tix[t]]) / NSIM for k in KEYS} for t in order}


def ko_path_for(ed, t, fr):
    labels = KO_LABELS[16]; path = []
    for rd in range(1, ed['nrd'] + 1):
        g = ed['tg'].get((t, rd))
        if g is None:
            break
        lbl = labels[rd - 1]; opp = g['b'] if g['a'] == t else g['a']
        if rd < fr:
            gf, ga = (g['hg'], g['ag']) if g['a'] == t else (g['ag'], g['hg'])
            step = dict(round=lbl, opp=opp, opp_flag=FLAG.get(opp, ''), gf=gf, ga=ga, won=(g['w'] == t))
            if g['so']:
                step['pens'] = True
            path.append(step)
            if g['w'] != t:
                break
        elif rd == fr:
            path.append(dict(round=lbl, opp=opp, opp_flag=FLAG.get(opp, ''), pending=True, date=str(g['date'])))
            break
        else:
            break
    return path


def build_snapshot(ed, fr):
    nrd = ed['nrd']; complete = fr > nrd
    asof = ed['last_grp'] if fr == 1 else ed['rend'][fr - 1]
    reach = simulate(ed['order'], ed['pwin'], asof, fr)
    alive_cnt = 1 if complete else 16 // (2 ** (fr - 1))
    teams = []
    for t in ed['order']:
        alive = (t == ed['champ']) if complete else ((t, fr) in ed['tg'])
        rr = reach[t]
        teams.append({
            'team': t, 'flag': FLAG.get(t, ''),
            'rating': round(rating_asof(t, asof) or 0.0, 2), 'rank': rank_asof(t, asof),
            'eliminated': (not alive),
            'champ': round(rr['champ'], 6), 'final': round(rr['final'], 6), 'sf': round(rr['sf'], 6),
            'qf': round(rr['qf'], 6), 'r16': round(rr['r16'], 6), 'r32': round(rr['r32'], 6),
            'ko_path': ko_path_for(ed, t, fr),
        })
    return {'date': str(asof), 'phase': 'knockout', 'games_left': max(0, alive_cnt - 1),
            'complete': complete, 'teams': teams}


# ── Build all editions ──────────────────────────────────────────────────────
buckets = []
for yr in range(1986, 2023, 4):
    ed = edition(yr)
    snaps = [build_snapshot(ed, fr) for fr in range(1, ed['nrd'] + 2)]   # R16..Champion
    buckets.append({'tournament': 'FIFA World Cup', 'edition': yr, 'snapshots': snaps})
    top = max(snaps[0]['teams'], key=lambda t: t['champ'])
    print(f"  {yr}: {len(snaps)} snapshots | pre-KO favorite {top['team']} {top['champ']*100:.0f}% | champ {ed['champ']}")

if WRITE:
    hist = json.load(open(f'{REPO}/docs/data/wc_odds_history.json'))
    existing = {(b['tournament'], b['edition']): b for b in hist['tournaments']}
    for b in buckets:
        existing[(b['tournament'], b['edition'])] = b          # replace/add retro editions
    hist['tournaments'] = sorted(existing.values(), key=lambda b: b['edition'], reverse=True)  # newest first
    json.dump(hist, open(f'{REPO}/docs/data/wc_odds_history.json', 'w'), separators=(',', ':'))
    print(f"WROTE {len(hist['tournaments'])} editions to wc_odds_history.json")
else:
    print("(dry run - pass --write to emit)")
