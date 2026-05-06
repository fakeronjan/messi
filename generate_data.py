"""
generate_data.py — reads messi_ratings_final.csv and writes JSON for the MESSI web frontend.
Run after messi.py. Outputs to docs/data/.

Mirrors the ZIDANE site architecture (multiple competitions / leagues), adapted for
international soccer (countries instead of clubs, tournaments instead of league seasons).
"""

import pandas as pd
import json
import os
import re
from bisect import bisect_right

os.makedirs('docs/data/teams', exist_ok=True)
os.makedirs('docs/data/seasons', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('messi_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date

games = pd.read_csv('all_soccer_games.csv')
games['date'] = pd.to_datetime(games['date']).dt.date

podiums = pd.read_csv('tournament_podiums.csv')


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# is_end_of_season: rows where year is even AND it's the team's latest snapshot in that year
# We compute this per team-year and tag the latest snapshot per even year.
df['is_end_of_season'] = 0
even_year_mask = (df['year'] % 2 == 0)
even_df = df[even_year_mask].copy()
# For each (country, year), find the max ranking_id (latest snapshot in that year)
year_max_id = even_df.groupby(['country', 'year'])['ranking_id'].transform('max')
df.loc[even_year_mask, 'is_end_of_season'] = (even_df['ranking_id'] == year_max_id).astype(int).values


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank_blend').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':              int(r['rank_blend']) if not pd.isna(r['rank_blend']) else None,
            'team':              r['country'],
            'confederation':     clean(r['confederation']),
            'rating':            round(float(r['rating_blend']), 3) if not pd.isna(r['rating_blend']) else None,
            'games_played':      int(r['games_played']) if not pd.isna(r['games_played']) else 0,
            'last_match':        clean(r['last_match']),
            'last_match_date':   clean(r['last_match_date']),
            'tournament_finish': clean(r['tournament_finish']),
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table (top 50 country-year ratings at end of even years) ─────────
print("Writing goat_teams.json...")
eos_all = df[df['is_end_of_season'] == 1].copy()
# Per country-year, take just one row (the EOS marker)
eos_top = (
    eos_all.dropna(subset=['rating_blend'])
    .sort_values('rating_blend', ascending=False)
    .head(50)
    .reset_index(drop=True)
)

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    goat_data.append({
        'rank':              i + 1,
        'team':              r['country'],
        'confederation':     clean(r['confederation']),
        'season':            int(r['year']),
        'rating':            round(float(r['rating_blend']), 3),
        'tournament_finish': clean(r['tournament_finish']),
    })
with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
# Per team: keep only game days + EOS markers (avoids gigantic files)
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
team_data = team_data.sort_values(['country', 'date'])

all_teams = sorted(df['country'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['country'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    confed = clean(tdf['confederation'].iloc[-1])
    teams_index.append({'name': team, 'confederation': confed, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('year'):
        if pd.isna(season):
            continue
        seasons[int(season)] = [
            {
                'date':              str(r['date']),
                'rating':            round(float(r['rating_blend']), 3) if not pd.isna(r['rating_blend']) else None,
                'rank':              int(r['rank_blend']) if not pd.isna(r['rank_blend']) else None,
                'last_match':        clean(r['last_match']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'is_game_day':       int(r['is_game_day']),
                'tournament_finish': clean(r['tournament_finish']),
            }
            for _, r in sdf.sort_values('date').iterrows()
        ]

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'confederation': confed, 'seasons': seasons},
                  f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files (one per year) ──────────────────────────────────
print("Writing season standings files...")
all_seasons = sorted(df['year'].dropna().unique())

for season in all_seasons:
    season = int(season)
    sdf = df[df['year'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank_blend')
        snap_date = str(rdf['date'].iloc[0])
        # Label: World Cup final day, end of even-year, or just date
        is_wc_final = int(rdf['is_world_cup_final_day'].iloc[0]) if 'is_world_cup_final_day' in rdf.columns else 0
        label = None
        if is_wc_final:
            label = 'World Cup Final Day'
        teams_snap = [
            {
                'rank':              int(r['rank_blend']) if not pd.isna(r['rank_blend']) else None,
                'team':              r['country'],
                'confederation':     clean(r['confederation']),
                'rating':            round(float(r['rating_blend']), 3) if not pd.isna(r['rating_blend']) else None,
                'last_match':        clean(r['last_match']),
                'last_match_date':   clean(r['last_match_date']),
                'tournament_finish': clean(r['tournament_finish']),
            }
            for _, r in rdf.iterrows()
        ]
        snapshots.append({'date': snap_date, 'label': label, 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(games['date'].min()),
    'last_date':  str(games['date'].max()),
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (per tournament) ──────────────────────────────────────
print("Writing champions.json...")

# Lookup country's rating + rank as of the latest snapshot in their tournament year.
# Build a (country, year) → (rating, rank) map.
year_eos = (
    df[df['is_end_of_season'] == 1]
    .groupby(['country', 'year'])[['rating_blend', 'rank_blend', 'confederation']]
    .first()
    .to_dict('index')
)


def country_eos_info(country, year):
    info = year_eos.get((country, int(year)))
    if not info:
        return {'rating': None, 'rank': None, 'confederation': ''}
    return {
        'rating': round(float(info['rating_blend']), 3) if not pd.isna(info['rating_blend']) else None,
        'rank':   int(info['rank_blend']) if not pd.isna(info['rank_blend']) else None,
        'confederation': clean(info['confederation']),
    }


# Group by tournament; for each, list editions newest-first
champions = {}
for tournament in sorted(podiums['tournament'].unique()):
    t_podiums = podiums[podiums['tournament'] == tournament]
    entries = []
    for year in sorted(t_podiums['year'].unique(), reverse=True):
        yp = t_podiums[t_podiums['year'] == year]
        first  = yp[yp['finish'] == 1]['team'].iloc[0] if len(yp[yp['finish'] == 1]) else None
        second = yp[yp['finish'] == 2]['team'].iloc[0] if len(yp[yp['finish'] == 2]) else None
        third  = yp[yp['finish'] == 3]['team'].iloc[0] if len(yp[yp['finish'] == 3]) else None

        def team_block(team_name):
            if not team_name:
                return None
            info = country_eos_info(team_name, year)
            return {
                'team':          team_name,
                'confederation': info['confederation'],
                'rating':        info['rating'],
                'rank':          info['rank'],
            }

        entries.append({
            'season':    int(year),
            'champion':  team_block(first),
            'runner_up': team_block(second),
            'third':     team_block(third),
        })
    champions[tournament] = entries

# Running championship counts per tournament (1st place only)
# Plus "all-tournaments" total per team
for tournament, entries in champions.items():
    counts = {}
    for entry in reversed(entries):
        if entry['champion']:
            ct = entry['champion']['team']
            counts[ct] = counts.get(ct, 0) + 1
            entry['champion']['title_count'] = counts[ct]

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
