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
from datetime import datetime, timezone
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


# ── Country → flag emoji map ──────────────────────────────────────────────────
# ISO 3166-1 alpha-2 codes converted to regional indicator pairs.
# Special UK tag-sequence flags handled inline. Defunct entities → ''.
_COUNTRY_TO_ISO = {
    # UEFA
    'Albania': 'AL', 'Andorra': 'AD', 'Armenia': 'AM', 'Austria': 'AT',
    'Azerbaijan': 'AZ', 'Belarus': 'BY', 'Belgium': 'BE', 'Bosnia and Herzegovina': 'BA',
    'Bulgaria': 'BG', 'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ',
    'Denmark': 'DK', 'Estonia': 'EE', 'Faroe Islands': 'FO', 'Finland': 'FI',
    'France': 'FR', 'Georgia': 'GE', 'Germany': 'DE', 'Gibraltar': 'GI',
    'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Israel': 'IL',
    'Italy': 'IT', 'Kazakhstan': 'KZ', 'Kosovo': 'XK', 'Latvia': 'LV',
    'Liechtenstein': 'LI', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Moldova': 'MD', 'Monaco': 'MC', 'Montenegro': 'ME', 'Netherlands': 'NL',
    'North Macedonia': 'MK', 'Norway': 'NO', 'Poland': 'PL',
    'Portugal': 'PT', 'Republic of Ireland': 'IE', 'Romania': 'RO', 'Russia': 'RU',
    'San Marino': 'SM', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI',
    'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR',
    'Ukraine': 'UA',
    # CONMEBOL
    'Argentina': 'AR', 'Bolivia': 'BO', 'Brazil': 'BR', 'Chile': 'CL',
    'Colombia': 'CO', 'Ecuador': 'EC', 'Paraguay': 'PY', 'Peru': 'PE',
    'Uruguay': 'UY', 'Venezuela': 'VE',
    # CONCACAF
    'Anguilla': 'AI', 'Antigua and Barbuda': 'AG', 'Aruba': 'AW',
    'Bahamas': 'BS', 'Barbados': 'BB', 'Belize': 'BZ', 'Bermuda': 'BM',
    'Bonaire': 'BQ', 'British Virgin Islands': 'VG', 'Canada': 'CA',
    'Cayman Islands': 'KY', 'Costa Rica': 'CR', 'Cuba': 'CU', 'Curaçao': 'CW',
    'Dominica': 'DM', 'Dominican Republic': 'DO', 'El Salvador': 'SV',
    'Grenada': 'GD', 'Guatemala': 'GT', 'Guyana': 'GY', 'Haiti': 'HT',
    'Honduras': 'HN', 'Jamaica': 'JM', 'Mexico': 'MX', 'Montserrat': 'MS',
    'Nicaragua': 'NI', 'Panama': 'PA', 'Puerto Rico': 'PR',
    'Saint Kitts and Nevis': 'KN', 'Saint Lucia': 'LC', 'Saint Martin': 'MF',
    'Saint Vincent and the Grenadines': 'VC', 'Sint Maarten': 'SX',
    'Suriname': 'SR', 'Trinidad and Tobago': 'TT',
    'Turks and Caicos Islands': 'TC', 'United States': 'US',
    'US Virgin Islands': 'VI', 'United States Virgin Islands': 'VI',
    'French Guiana': 'GF', 'Guadeloupe': 'GP', 'Martinique': 'MQ',
    # CAF
    'Algeria': 'DZ', 'Angola': 'AO', 'Benin': 'BJ', 'Botswana': 'BW',
    'Burkina Faso': 'BF', 'Burundi': 'BI', 'Cameroon': 'CM', 'Cape Verde': 'CV',
    'Central African Republic': 'CF', 'Chad': 'TD', 'Comoros': 'KM',
    'DR Congo': 'CD', 'Congo': 'CG', 'Djibouti': 'DJ', 'Egypt': 'EG',
    'Equatorial Guinea': 'GQ', 'Eritrea': 'ER', 'Eswatini': 'SZ',
    'Ethiopia': 'ET', 'Gabon': 'GA', 'Gambia': 'GM', 'Ghana': 'GH',
    'Guinea': 'GN', 'Guinea-Bissau': 'GW', 'Ivory Coast': 'CI', 'Kenya': 'KE',
    'Lesotho': 'LS', 'Liberia': 'LR', 'Libya': 'LY', 'Madagascar': 'MG',
    'Malawi': 'MW', 'Mali': 'ML', 'Mauritania': 'MR', 'Mauritius': 'MU',
    'Morocco': 'MA', 'Mozambique': 'MZ', 'Namibia': 'NA', 'Niger': 'NE',
    'Nigeria': 'NG', 'Rwanda': 'RW', 'São Tomé and Príncipe': 'ST',
    'Senegal': 'SN', 'Seychelles': 'SC', 'Sierra Leone': 'SL', 'Somalia': 'SO',
    'South Africa': 'ZA', 'South Sudan': 'SS', 'Sudan': 'SD', 'Tanzania': 'TZ',
    'Togo': 'TG', 'Tunisia': 'TN', 'Uganda': 'UG', 'Zambia': 'ZM',
    'Zimbabwe': 'ZW', 'Zanzibar': '',  # Zanzibar — no flag emoji
    # AFC
    'Afghanistan': 'AF', 'Australia': 'AU', 'Bahrain': 'BH', 'Bangladesh': 'BD',
    'Bhutan': 'BT', 'Brunei': 'BN', 'Cambodia': 'KH', 'China': 'CN',
    'China PR': 'CN', 'Chinese Taipei': 'TW', 'Taiwan': 'TW', 'Guam': 'GU',
    'Hong Kong': 'HK', 'India': 'IN', 'Indonesia': 'ID', 'Iran': 'IR',
    'Iraq': 'IQ', 'Japan': 'JP', 'Jordan': 'JO', 'Kuwait': 'KW',
    'Kyrgyzstan': 'KG', 'Laos': 'LA', 'Lebanon': 'LB', 'Macau': 'MO',
    'Malaysia': 'MY', 'Maldives': 'MV', 'Mongolia': 'MN', 'Myanmar': 'MM',
    'Nepal': 'NP', 'North Korea': 'KP', 'Northern Mariana Islands': 'MP',
    'Oman': 'OM', 'Pakistan': 'PK', 'Palestine': 'PS', 'Philippines': 'PH',
    'Qatar': 'QA', 'Saudi Arabia': 'SA', 'Singapore': 'SG', 'South Korea': 'KR',
    'Sri Lanka': 'LK', 'Syria': 'SY', 'Tajikistan': 'TJ', 'Thailand': 'TH',
    'Timor-Leste': 'TL', 'Turkmenistan': 'TM', 'United Arab Emirates': 'AE',
    'Uzbekistan': 'UZ', 'Vietnam': 'VN', 'Yemen': 'YE', 'Yemen DPR': '',
    # OFC
    'American Samoa': 'AS', 'Cook Islands': 'CK', 'Fiji': 'FJ',
    'New Caledonia': 'NC', 'New Zealand': 'NZ', 'Papua New Guinea': 'PG',
    'Samoa': 'WS', 'Solomon Islands': 'SB', 'Tahiti': 'PF', 'Tonga': 'TO',
    'Tuvalu': 'TV', 'Vanuatu': 'VU',
}

# UK constituent countries use special tag sequences
_UK_FLAGS = {
    'England':          '🏴\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F',
    'Scotland':         '🏴\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F',
    'Wales':            '🏴\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F',
    'Northern Ireland': '🇬🇧',  # No tag sequence; fall back to UK flag
}

# Defunct entities — no flag (intentional)
_DEFUNCT = {
    'Soviet Union', 'Yugoslavia', 'FR Yugoslavia', 'Serbia and Montenegro',
    'Czechoslovakia', 'East Germany', 'West Germany', 'German DR',
}


def country_flag(country):
    if country in _UK_FLAGS:
        return _UK_FLAGS[country]
    if country in _DEFUNCT:
        return ''
    code = _COUNTRY_TO_ISO.get(country)
    if not code or len(code) != 2:
        return ''
    # Regional indicator: A = U+1F1E6
    return chr(0x1F1E6 + ord(code[0]) - ord('A')) + chr(0x1F1E6 + ord(code[1]) - ord('A'))


# Tournament name → short label (for honor badges)
_TOURNAMENT_ABBREV = {
    'FIFA World Cup':           'WC',
    'UEFA Euro':                'Euro',
    'Copa América':             'Copa',
    'AFC Asian Cup':            'Asian Cup',
    'UEFA Nations League':      'UNL',
    'CONCACAF Nations League':  'CNL',
}


# Host country flags per tournament edition. Co-hosted = list of flags.
# Empty list for editions without a single clear host (e.g. UEFA Euro 2020 pan-European).
_TOURNAMENT_HOSTS = {
    ('FIFA World Cup', 1982): ['Spain'],
    ('FIFA World Cup', 1986): ['Mexico'],
    ('FIFA World Cup', 1990): ['Italy'],
    ('FIFA World Cup', 1994): ['United States'],
    ('FIFA World Cup', 1998): ['France'],
    ('FIFA World Cup', 2002): ['Japan', 'South Korea'],
    ('FIFA World Cup', 2006): ['Germany'],
    ('FIFA World Cup', 2010): ['South Africa'],
    ('FIFA World Cup', 2014): ['Brazil'],
    ('FIFA World Cup', 2018): ['Russia'],
    ('FIFA World Cup', 2022): ['Qatar'],

    ('UEFA Euro', 1980): ['Italy'],
    ('UEFA Euro', 1984): ['France'],
    ('UEFA Euro', 1988): ['Germany'],   # West Germany at the time
    ('UEFA Euro', 1992): ['Sweden'],
    ('UEFA Euro', 1996): ['England'],
    ('UEFA Euro', 2000): ['Belgium', 'Netherlands'],
    ('UEFA Euro', 2004): ['Portugal'],
    ('UEFA Euro', 2008): ['Austria', 'Switzerland'],
    ('UEFA Euro', 2012): ['Poland', 'Ukraine'],
    ('UEFA Euro', 2016): ['France'],
    ('UEFA Euro', 2020): [],  # Pan-European, 11 cities
    ('UEFA Euro', 2024): ['Germany'],

    ('Copa América', 1987): ['Argentina'],
    ('Copa América', 1989): ['Brazil'],
    ('Copa América', 1991): ['Chile'],
    ('Copa América', 1993): ['Ecuador'],
    ('Copa América', 1995): ['Uruguay'],
    ('Copa América', 1997): ['Bolivia'],
    ('Copa América', 1999): ['Paraguay'],
    ('Copa América', 2001): ['Colombia'],
    ('Copa América', 2004): ['Peru'],
    ('Copa América', 2007): ['Venezuela'],
    ('Copa América', 2011): ['Argentina'],
    ('Copa América', 2015): ['Chile'],
    ('Copa América', 2016): ['United States'],   # Centenario
    ('Copa América', 2019): ['Brazil'],
    ('Copa América', 2021): ['Brazil'],
    ('Copa América', 2024): ['United States'],

    ('AFC Asian Cup', 1980): ['Kuwait'],
    ('AFC Asian Cup', 1984): ['Singapore'],
    ('AFC Asian Cup', 1988): ['Qatar'],
    ('AFC Asian Cup', 1992): ['Japan'],
    ('AFC Asian Cup', 1996): ['United Arab Emirates'],
    ('AFC Asian Cup', 2000): ['Lebanon'],
    ('AFC Asian Cup', 2004): ['China PR'],
    ('AFC Asian Cup', 2007): ['Indonesia', 'Malaysia', 'Thailand', 'Vietnam'],
    ('AFC Asian Cup', 2011): ['Qatar'],
    ('AFC Asian Cup', 2015): ['Australia'],
    ('AFC Asian Cup', 2019): ['United Arab Emirates'],
    ('AFC Asian Cup', 2024): ['Qatar'],
}


def host_flags(tournament, year):
    hosts = _TOURNAMENT_HOSTS.get((tournament, int(year)), [])
    return ''.join(country_flag(h) for h in hosts if country_flag(h))


# Per-(country, year) → list of tournament finishes
# Each entry: {tournament_short, finish} — used for honor badges with tournament context
_country_year_finishes = {}
for _, p in podiums.iterrows():
    key = (p['team'], int(p['year']))
    _country_year_finishes.setdefault(key, []).append({
        'tournament': _TOURNAMENT_ABBREV.get(p['tournament'], p['tournament']),
        'finish': int(p['finish']),
    })
# Sort each list: best finish first
for key in _country_year_finishes:
    _country_year_finishes[key].sort(key=lambda x: x['finish'])


def country_year_finishes(country, year):
    if pd.isna(year):
        return []
    return _country_year_finishes.get((country, int(year)), [])


# is_end_of_season: rows whose snapshot date falls on the LAST DAY of a major tournament.
# Captures each team's rating at peak tournament moments (WC final day, Euro
# final day, etc.) rather than fading-into-year-end snapshots after late-year
# friendlies. Names match the source data (see messi.py PODIUM_TOURNAMENTS).
PODIUM_TOURNAMENTS = [
    'FIFA World Cup',
    'UEFA Euro',
    'Copa América',
    'African Cup of Nations',
    'AFC Asian Cup',
    'Gold Cup',
    'Oceania Nations Cup',
    'UEFA Nations League',
    'CONCACAF Nations League',
    'FIFA Confederations Cup',
]
# Per-tournament + per-year final date map. Used both for is_end_of_season
# (set of all final dates) and the GOAT anchor logic (specific date per
# (tournament, year)).
_tournament_final_date_map = {}  # (tournament, year_int) -> final pd.Timestamp
for _t in PODIUM_TOURNAMENTS:
    _tg = games[games['tournament'] == _t]
    if _tg.empty:
        continue
    for _year, _grp in _tg.groupby(_tg['date'].apply(lambda d: d.year)):
        _tournament_final_date_map[(_t, int(_year))] = _grp['date'].max()
_tournament_final_dates = set(_tournament_final_date_map.values())

df['is_end_of_season'] = df['date'].apply(lambda d: 1 if d in _tournament_final_dates else 0)
print(f"Tournament-end snapshot dates: {len(_tournament_final_dates)}")

# Confederation rank within each ranking_id snapshot
df['conf_rank'] = df.groupby(['ranking_id', 'confederation'])['rating'] \
                    .rank(method='min', ascending=False)

# Continental winners: (team, year) tuples for any non-WC tournament won.
# Used for the gold-pill flag on the Conf column in Team Summary.
_CONTINENTAL_TOURNAMENTS = {
    'UEFA Euro', 'Copa América', 'AFC Asian Cup', 'African Cup of Nations',
    'Gold Cup', 'Oceania Nations Cup', 'UEFA Nations League', 'CONCACAF Nations League',
}
_continental_winners = set(
    (r['team'], int(r['year']))
    for _, r in podiums.iterrows()
    if r['tournament'] in _CONTINENTAL_TOURNAMENTS and r['finish'] == 1
)


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':                int(r['rank']) if not pd.isna(r['rank']) else None,
            'team':                r['country'],
            'flag':                country_flag(r['country']),
            'confederation':       clean(r['confederation']),
            'rating':              round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
            'games_played':        int(r['games_played']) if not pd.isna(r['games_played']) else 0,
            'last_match':          clean(r['last_match']),
            'last_match_date':     clean(r['last_match_date']),
            'tournament_finishes': country_year_finishes(r['country'], r['year']),
            'continental_winner':  1 if (r['country'], int(r['year'])) in _continental_winners else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table (top 50 country-year ratings at end of major tournaments) ─
# Eligibility: medaled (1st, 2nd, or 3rd) in any major tournament that year.
# Each team is then anchored at the SPECIFIC tournament's final date (the
# moment they actually finished on the podium) — not whichever snapshot in
# the year had the highest rating. If a team medaled in multiple tournaments
# in one year, we use the rating from whichever is highest.
#
# Also requires games_played >= 6 in the rolling window — filters teams
# whose Massey rating comes from too small a sample of weak-confederation
# opponents (e.g. Tahiti 2012 OFC only had 5 OFC qualifier games).
print("Writing goat_teams.json...")
podiums = pd.read_csv('tournament_podiums.csv')
GOAT_MIN_GAMES = 6

# Medalist gate: 1st, 2nd, or 3rd in any major tournament. Matches the
# cross-site GOAT convention established 2026-05-11 — every other site
# filters its GOAT to "teams that contested for the championship" and this
# is the soccer-international equivalent (medal = reached knockout rounds).
_eligible_rows = (
    podiums[podiums['finish'].isin([1, 2, 3])][['team', 'year', 'tournament']]
    .drop_duplicates()
)

# For each eligible row, pull the team's rating at THAT tournament's final
# date. Skip if games_played < GOAT_MIN_GAMES at the snapshot.
df_indexed = df.set_index(['country', 'date'])
candidate_records = []
for _, row in _eligible_rows.iterrows():
    final_date = _tournament_final_date_map.get((row['tournament'], int(row['year'])))
    if final_date is None:
        continue
    # df dates are pd.Timestamp.date() objects already (set at file load).
    # _tournament_final_date_map values come from games['date'] which can be
    # either pd.Timestamp or datetime.date depending on dtype — normalize.
    final_date_key = final_date.date() if hasattr(final_date, 'date') else final_date
    try:
        snap = df_indexed.loc[(row['team'], final_date_key)]
    except KeyError:
        continue
    if isinstance(snap, pd.DataFrame):
        snap = snap.iloc[0]
    if pd.isna(snap.get('rating')):
        continue
    if snap.get('games_played', 0) < GOAT_MIN_GAMES:
        continue
    candidate_records.append({
        'country':       row['team'],
        'year':          int(row['year']),
        'tournament':    row['tournament'],
        'rating':  snap['rating'],
        'rank':    snap.get('rank'),
        'confederation': snap.get('confederation', ''),
        'games_played':  int(snap.get('games_played', 0)),
        'date':          final_date,
    })

eos_top = (
    pd.DataFrame(candidate_records)
    .sort_values('rating', ascending=False)
    .drop_duplicates(subset=['country', 'year'], keep='first')  # keep highest if multi-tournament year
    .head(50)
    .reset_index(drop=True)
)

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    goat_data.append({
        'rank':                i + 1,
        'team':                r['country'],
        'flag':                country_flag(r['country']),
        'confederation':       clean(r['confederation']),
        'season':              int(r['year']),
        'rating':              round(float(r['rating']), 3),
        'tournament_finishes': country_year_finishes(r['country'], r['year']),
        'continental_winner':  1 if (r['country'], int(r['year'])) in _continental_winners else 0,
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
    flag = country_flag(team)
    teams_index.append({'name': team, 'flag': flag, 'confederation': confed, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('year'):
        if pd.isna(season):
            continue
        finishes_for_year = country_year_finishes(team, season)
        won_continental = (team, int(season)) in _continental_winners
        seasons[int(season)] = [
            {
                'date':                str(r['date']),
                'rating':              round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
                'rank':                int(r['rank']) if not pd.isna(r['rank']) else None,
                'conf_rank':           int(r['conf_rank']) if not pd.isna(r['conf_rank']) else None,
                'last_match':          clean(r['last_match']),
                'is_end_of_season':    int(r['is_end_of_season']),
                'is_game_day':         int(r['is_game_day']),
                'tournament_finishes': finishes_for_year,
                'continental_winner':  1 if won_continental else 0,
            }
            for _, r in sdf.sort_values('date').iterrows()
        ]

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'flag': flag, 'confederation': confed, 'seasons': seasons},
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
        rdf = rdf.sort_values('rank')
        snap_date = str(rdf['date'].iloc[0])
        # Label: World Cup final day, end of even-year, or just date
        is_wc_final = int(rdf['is_world_cup_final_day'].iloc[0]) if 'is_world_cup_final_day' in rdf.columns else 0
        label = None
        if is_wc_final:
            label = 'World Cup Final Day'
        teams_snap = [
            {
                'rank':                int(r['rank']) if not pd.isna(r['rank']) else None,
                'team':                r['country'],
                'flag':                country_flag(r['country']),
                'confederation':       clean(r['confederation']),
                'rating':              round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
                'last_match':          clean(r['last_match']),
                'last_match_date':     clean(r['last_match_date']),
                'tournament_finishes': country_year_finishes(r['country'], r['year']),
                'continental_winner':  1 if (r['country'], int(r['year'])) in _continental_winners else 0,
            }
            for _, r in rdf.iterrows()
        ]
        snapshots.append({'date': snap_date, 'label': label, 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    # Use the rated period, not the raw games period:
    # - first_date excludes pre-1986 warm-up (script filters out earlier ratings)
    # - last_date excludes future-scheduled fixtures present in the source data
    'first_date': str(df['date'].min()),
    'last_date':  str(df['date'].max()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (per tournament) ──────────────────────────────────────
print("Writing champions.json...")

# Per-tournament final-day lookup: for each podium tournament + year, find
# the date of its FINAL match. Champions table needs to read each team's
# rating as of THAT specific tournament's final day — not whichever EOS row
# happens to come first in the year. (Previously a country with multiple
# podium tournaments in one year would get the earliest EOS rating, which
# wildly understated e.g. France's 2022 WC rating because the CONCACAF
# Nations League finished in June 2022 before France's WC run started.)
games['date'] = pd.to_datetime(games['date'])
_tournament_final_date_by_yt = {}
for _t in PODIUM_TOURNAMENTS:
    _tg = games[games['tournament'] == _t]
    if _tg.empty:
        continue
    for _y, _grp in _tg.groupby(_tg['date'].apply(lambda d: d.year)):
        # Normalize to datetime.date so comparisons against df['date'] (also
        # datetime.date from line 20) succeed. Without this, every champion
        # entry's rating/rank renders null because Timestamp != date.
        _tournament_final_date_by_yt[(_t, int(_y))] = _grp['date'].max().date()


def country_tournament_info(country, tournament, year):
    """Look up a country's rating + rank as of a specific tournament's final day."""
    final_date = _tournament_final_date_by_yt.get((tournament, int(year)))
    if final_date is None:
        return {'rating': None, 'rank': None, 'conf_rank': None, 'confederation': ''}
    rows = df[(df['country'] == country) & (df['date'] == final_date)]
    if rows.empty:
        return {'rating': None, 'rank': None, 'conf_rank': None, 'confederation': ''}
    r = rows.iloc[0]
    return {
        'rating':        round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
        'rank':          int(r['rank']) if not pd.isna(r['rank']) else None,
        'conf_rank':     int(r['conf_rank']) if not pd.isna(r['conf_rank']) else None,
        'confederation': clean(r['confederation']),
    }


# Disputed / administratively-awarded titles. Each entry surfaces a small
# "(disputed)" tag in the Tournaments tab next to that team's listing.
# Format: (tournament, year, finish_pos, team) → short reason string.
DISPUTED_TITLES = {
    ('African Cup of Nations', 2026, 1, 'Morocco'):
        'Awarded by CAF in March 2026 after Senegal walk-off; Senegal originally won 1-0 on the field.',
    ('African Cup of Nations', 2026, 2, 'Senegal'):
        'Original on-field winner (1-0 vs Morocco); title stripped by CAF in March 2026.',
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

        def team_block(team_name, finish_pos, _tour=tournament, _yr=year):
            if not team_name:
                return None
            info = country_tournament_info(team_name, _tour, _yr)
            block = {
                'team':          team_name,
                'flag':          country_flag(team_name),
                'confederation': info['confederation'],
                'rating':        info['rating'],
                'rank':          info['rank'],
                'conf_rank':     info['conf_rank'],
            }
            disputed = DISPUTED_TITLES.get((_tour, _yr, finish_pos, team_name))
            if disputed:
                block['disputed'] = disputed
            return block

        entries.append({
            'season':     int(year),
            'host_flags': host_flags(tournament, year),
            'champion':   team_block(first,  1),
            'runner_up':  team_block(second, 2),
            'third':      team_block(third,  3),
        })
    champions[tournament] = entries

# Pre-data tournament finishes — seed running counters with results from
# editions before our rated dataset begins (FIFA WC 1930-1978 not in our data,
# Euro 1960-1976, etc.). Keys use team names as they appear in our data.
# Defunct teams (Soviet Union, Yugoslavia, etc.) get seeded too — they just
# never accumulate further counts since they don't appear in post-data entries.
PRE_DATA_TOURNAMENT_COUNTS = {
    'FIFA World Cup': {
        'champion': {  # 1930-1978
            'Brazil': 3, 'Italy': 2, 'West Germany': 2, 'Uruguay': 2,
            'Argentina': 1, 'England': 1,
        },
        'runner_up': {
            'Hungary': 2, 'Czechoslovakia': 2, 'Netherlands': 2,
            'Argentina': 1, 'Brazil': 1, 'Sweden': 1, 'West Germany': 1, 'Italy': 1,
        },
        'third': {
            'Brazil': 2,
            'United States': 1, 'Sweden': 1, 'Austria': 1, 'France': 1,
            'Chile': 1, 'Portugal': 1, 'West Germany': 1, 'Poland': 1,
        },
    },
    'UEFA Euro': {
        'champion': {  # 1960-1976
            'Spain': 1, 'Italy': 1, 'West Germany': 1, 'Soviet Union': 1, 'Czechoslovakia': 1,
        },
        'runner_up': {
            'Yugoslavia': 2, 'Soviet Union': 2, 'West Germany': 1,
        },
        'third': {  # 3rd place playoffs existed 1960-1980
            'Czechoslovakia': 1, 'Hungary': 1, 'England': 1, 'Belgium': 1, 'Netherlands': 1,
        },
    },
    'Copa América': {
        'champion': {  # 1916-1983 (16 editions of varying frequency)
            'Uruguay': 13, 'Argentina': 12, 'Brazil': 3, 'Paraguay': 2, 'Peru': 2, 'Bolivia': 1,
        },
        # Runner-ups and 3rd places not seeded — sparse pre-1987 records
    },
    'AFC Asian Cup': {
        'champion': {  # 1956-1976
            'Iran': 3, 'South Korea': 2, 'Israel': 1,
        },
        'runner_up': {
            'Israel': 2, 'India': 1, 'Myanmar': 1, 'South Korea': 1, 'Kuwait': 1,
        },
        # 3rd places pre-1980 not seeded
    },
    # Nations Leagues started in 2018/2019 — no pre-data history
}


# Running counts per tournament (champion / runner-up / third), seeded with pre-data totals
for tournament, entries in champions.items():
    seeds = PRE_DATA_TOURNAMENT_COUNTS.get(tournament, {})
    champ_counts = dict(seeds.get('champion', {}))
    ru_counts    = dict(seeds.get('runner_up', {}))
    third_counts = dict(seeds.get('third', {}))
    for entry in reversed(entries):
        if entry['champion']:
            ct = entry['champion']['team']
            champ_counts[ct] = champ_counts.get(ct, 0) + 1
            entry['champion']['title_count'] = champ_counts[ct]
        if entry['runner_up']:
            rt = entry['runner_up']['team']
            ru_counts[rt] = ru_counts.get(rt, 0) + 1
            entry['runner_up']['runner_up_count'] = ru_counts[rt]
        if entry['third']:
            tt = entry['third']['team']
            third_counts[tt] = third_counts.get(tt, 0) + 1
            entry['third']['third_count'] = third_counts[tt]

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
