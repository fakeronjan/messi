"""
generate_data.py — reads messi_ratings_final.csv.gz and writes JSON for the MESSI web frontend.
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
df = pd.read_csv('messi_ratings_final.csv.gz')
df['date'] = pd.to_datetime(df['date']).dt.date

games = pd.read_csv('all_soccer_games.csv')
games['date'] = pd.to_datetime(games['date']).dt.date

podiums = pd.read_csv('tournament_podiums.csv')


# DILLON era-aware naming (user-locked 2026-05-29). MESSI works in name-space
# (the engine rates by country name), so NAME_HISTORY is keyed by canonical
# display name. Each entry: (historical_name, start_iso, end_iso). MESSI
# already silently merges these renames at scrape time -- this just makes
# the era visible inline via display_name on each per-row entry.
NAME_HISTORY_BY_NAME = {
    'Russia':    [('Soviet Union', '1900-01-01', '1991-12-25')],
    'Germany':   [('West Germany',  '1900-01-01', '1990-10-03')],
    'Myanmar':   [('Burma',         '1900-01-01', '1989-06-18')],
    'DR Congo':  [('Zaire',         '1971-10-27', '1997-05-17')],
    'Sri Lanka': [('Ceylon',        '1900-01-01', '1972-05-22')],
    'Samoa':     [('Western Samoa', '1900-01-01', '1997-07-04')],
}
_NAME_HISTORY_PARSED = {
    n: [(hn, pd.to_datetime(s).date(), pd.to_datetime(e).date())
        for hn, s, e in entries]
    for n, entries in NAME_HISTORY_BY_NAME.items()
}
HISTORICAL_NAMES_BY_NAME = {
    n: [hn for hn, _, _ in entries] for n, entries in NAME_HISTORY_BY_NAME.items()
}


def display_name_at(country, as_of):
    """Era-aware display name. Returns the historical name in effect on
    `as_of` if NAME_HISTORY_BY_NAME covers it; else None."""
    hist = _NAME_HISTORY_PARSED.get(country)
    if not hist:
        return None
    d = as_of if hasattr(as_of, 'year') else pd.to_datetime(as_of).date()
    for name, start, end in hist:
        if start <= d <= end:
            return name
    return None


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
    ('FIFA World Cup', 1930): ['Uruguay'],
    ('FIFA World Cup', 1934): ['Italy'],
    ('FIFA World Cup', 1938): ['France'],
    ('FIFA World Cup', 1950): ['Brazil'],
    ('FIFA World Cup', 1954): ['Switzerland'],
    ('FIFA World Cup', 1958): ['Sweden'],
    ('FIFA World Cup', 1962): ['Chile'],
    ('FIFA World Cup', 1966): ['England'],
    ('FIFA World Cup', 1970): ['Mexico'],
    ('FIFA World Cup', 1974): ['Germany'],   # West Germany at the time
    ('FIFA World Cup', 1978): ['Argentina'],
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
# In-progress gate (see [[feedback-in-progress-season-gate]]): only assign a
# "final date" to a tournament edition that has actually concluded. MESSI
# games carry no `round` field, so the only completion signal here is a
# curated podium entry. Without this gate, an in-progress event's most
# recent played game-day gets labeled the "Final", which is wrong.
_podium_editions = set(zip(podiums['tournament'], podiums['year'].astype(int)))
_tournament_final_date_map = {}  # (tournament, year_int) -> final pd.Timestamp
for _t in PODIUM_TOURNAMENTS:
    _tg = games[games['tournament'] == _t]
    if _tg.empty:
        continue
    for _year, _grp in _tg.groupby(_tg['date'].apply(lambda d: d.year)):
        _year = int(_year)
        if (_t, _year) not in _podium_editions:
            continue  # in progress -- no Final label assigned
        _tournament_final_date_map[(_t, _year)] = _grp['date'].max()
_tournament_final_dates = set(_tournament_final_date_map.values())

df['is_end_of_season'] = df['date'].apply(lambda d: 1 if d in _tournament_final_dates else 0)
print(f"Tournament-end snapshot dates: {len(_tournament_final_dates)}")

# Per-(team, year) anchor for the Team Summary cross-season view: pick ONE
# representative snapshot per team-year. Priority:
#   1. FIFA World Cup Final day (if that year had one)
#   2. Team's confederation continental championship final day
#   3. Other major tournament the team participated in (latest final date)
#   4. Fallback: last game-day of the year for that team
# Output: `_team_year_anchor_date[(team, year)] -> (Timestamp, label)`.
_CONFED_CHAMPIONSHIP = {
    'UEFA':       'UEFA Euro',
    'CONMEBOL':   'Copa América',
    'AFC':        'AFC Asian Cup',
    'CAF':        'African Cup of Nations',
    'CONCACAF':   'Gold Cup',
    'OFC':        'Oceania Nations Cup',
}
_OTHER_TOURNAMENTS = [  # tried after WC + continental, in this priority order
    'UEFA Nations League', 'CONCACAF Nations League', 'FIFA Confederations Cup',
]
# (team, year) -> set of tournaments participated in
_team_year_tournaments = {}
for _t in PODIUM_TOURNAMENTS:
    _tg = games[games['tournament'] == _t]
    for _, _g in _tg.iterrows():
        _yr = _g['date'].year
        for _team in (_g['home_team'], _g['away_team']):
            _team_year_tournaments.setdefault((_team, _yr), set()).add(_t)

# (team, year) -> team confederation (latest known)
_team_confed = (
    df.dropna(subset=['confederation'])
      .groupby('country')['confederation'].last().to_dict()
)
# (team, year) -> last actual game-day Timestamp (fallback for years with no major tournament)
_team_year_last_game = (
    df[df['is_game_day'] == 1]
      .dropna(subset=['year'])
      .groupby(['country', 'year'])['date'].max().to_dict()
)

_team_year_anchor = {}  # (team, year_int) -> (Timestamp, label)
for (team, year_f), last_game in _team_year_last_game.items():
    year = int(year_f)
    played = _team_year_tournaments.get((team, year), set())
    chosen = None  # (Timestamp, label)
    # 1. World Cup
    if 'FIFA World Cup' in played:
        d = _tournament_final_date_map.get(('FIFA World Cup', year))
        if d is not None: chosen = (d, 'End of FIFA World Cup')
    # 2. Confederation championship
    if chosen is None:
        confed = _team_confed.get(team)
        confed_t = _CONFED_CHAMPIONSHIP.get(confed)
        if confed_t and confed_t in played:
            d = _tournament_final_date_map.get((confed_t, year))
            if d is not None: chosen = (d, f'End of {confed_t}')
    # 3. Other major tournament
    if chosen is None:
        for _t in _OTHER_TOURNAMENTS:
            if _t in played:
                d = _tournament_final_date_map.get((_t, year))
                if d is not None:
                    chosen = (d, f'End of {_t}')
                    break
    # 4. Fallback: last game-day of the year
    if chosen is None:
        chosen = (last_game, 'End of year')
    _team_year_anchor[(team, year)] = chosen

# Per-row flag + label string for the year-anchor row of each team-year
df['is_year_anchor'] = 0
df['year_anchor_label'] = ''
_anchor_rows_idx = []
for (team, year), (d, label) in _team_year_anchor.items():
    mask = (df['country'] == team) & (df['date'] == d) & (df['year'] == year)
    if mask.any():
        df.loc[mask, 'is_year_anchor'] = 1
        df.loc[mask, 'year_anchor_label'] = label
print(f"Year-anchor flagged rows: {(df['is_year_anchor']==1).sum():,} (one per team-year where data exists)")

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
# Build the lookup index using string dates to avoid pandas-version-dependent
# type-handling differences (pandas 2.x normalizes datetime.date in index to
# Timestamps, which then fails .loc on a date-key lookup).
_df_with_str_date = df.copy()
_df_with_str_date['_date_str'] = _df_with_str_date['date'].astype(str)
df_indexed = _df_with_str_date.set_index(['country', '_date_str'])
candidate_records = []
for _, row in _eligible_rows.iterrows():
    final_date = _tournament_final_date_map.get((row['tournament'], int(row['year'])))
    if final_date is None:
        continue
    final_date_key = str(final_date.date() if hasattr(final_date, 'date') else final_date)
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

print(f"  GOAT candidate_records: {len(candidate_records)} (from {len(_eligible_rows)} eligible podium rows)")
if not candidate_records:
    # Defensive: empty candidates would crash sort_values. Emit empty goat_teams.json
    # and a diagnostic so we can investigate the underlying lookup failure.
    print("  WARNING: candidate_records is empty. Writing empty goat_teams.json.")
    print(f"  Diagnostic: df shape={df.shape}, df.columns={list(df.columns)}")
    print(f"  Sample df['date'] dtypes: {df['date'].head(3).tolist()}")
    print(f"  Sample _tournament_final_date_map values: {list(_tournament_final_date_map.values())[:3]}")
    with open('docs/data/goat_teams.json', 'w') as f:
        json.dump([], f)
    eos_top = pd.DataFrame(columns=['country', 'year', 'tournament', 'rating', 'rank',
                                     'confederation', 'games_played', 'date'])
else:
    eos_top = (
        pd.DataFrame(candidate_records)
        .sort_values('rating', ascending=False)
        .drop_duplicates(subset=['country', 'year'], keep='first')  # keep highest if multi-tournament year
        .head(50)
        .reset_index(drop=True)
    )

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    entry = {
        'rank':                i + 1,
        'team':                r['country'],
        'flag':                country_flag(r['country']),
        'confederation':       clean(r['confederation']),
        'season':              int(r['year']),
        'rating':              round(float(r['rating']), 3),
        'tournament_finishes': country_year_finishes(r['country'], r['year']),
        'continental_winner':  1 if (r['country'], int(r['year'])) in _continental_winners else 0,
    }
    era = display_name_at(r['country'], r['date'])
    if era and era != r['country']:
        entry['display_name'] = era
    goat_data.append(entry)
with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
# Per team: keep only game days + EOS markers (avoids gigantic files)
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1) | (df['is_year_anchor'] == 1)].copy()
team_data = team_data.sort_values(['country', 'date'])

# (team, year) set of nations that actually played ≥1 game that year. Used to
# drop "ghost" year entries — without this, defunct nations (Czechoslovakia
# 1994 etc.) and live-but-inactive nations (Mongolia 2025-26 etc.) get a
# carried-over year-end snapshot from the rolling window even though they
# played zero games that year.
played_team_years = set(
    (str(t), int(y)) for t, y in
    df.loc[df['is_game_day'] == 1, ['country', 'year']]
      .dropna().itertuples(index=False, name=None)
)

all_teams = sorted(df['country'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['country'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    confed = clean(tdf['confederation'].iloc[-1])
    flag = country_flag(team)
    hist_names = HISTORICAL_NAMES_BY_NAME.get(team, [])
    idx_entry = {'name': team, 'flag': flag, 'confederation': confed, 'slug': team_slug}
    if hist_names:
        idx_entry['historical_names'] = hist_names
    teams_index.append(idx_entry)

    seasons = {}
    for season, sdf in tdf.groupby('year'):
        if pd.isna(season):
            continue
        if (str(team), int(season)) not in played_team_years:
            continue  # skip ghost year — team played 0 games
        finishes_for_year = country_year_finishes(team, season)
        won_continental = (team, int(season)) in _continental_winners
        rows = []
        for _, r in sdf.sort_values('date').iterrows():
            row = {
                'date':                str(r['date']),
                'rating':              round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
                'rank':                int(r['rank']) if not pd.isna(r['rank']) else None,
                'conf_rank':           int(r['conf_rank']) if not pd.isna(r['conf_rank']) else None,
                'last_match':          clean(r['last_match']),
                'is_end_of_season':    int(r['is_end_of_season']),
                'is_game_day':         int(r['is_game_day']),
                'is_year_anchor':      int(r.get('is_year_anchor', 0) or 0),
                'year_anchor_label':   clean(r.get('year_anchor_label', '')),
                'tournament_finishes': finishes_for_year,
                'continental_winner':  1 if won_continental else 0,
            }
            era = display_name_at(team, r['date'])
            if era and era != team:
                row['display_name'] = era
            rows.append(row)
        seasons[int(season)] = rows

    team_doc = {'team': team, 'flag': flag, 'confederation': confed, 'seasons': seasons}
    if hist_names:
        team_doc['historical_names'] = hist_names
    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump(team_doc, f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files (one per year) ──────────────────────────────────
# Tournament-end labels: map a date to (label, prestige) for major tournaments.
# Prestige (lower = more prestigious) controls dropdown default + ordering.
# Edition detection uses gap-based segmentation (>= 90-day gap = new edition)
# so editions straddling calendar years (e.g., ACoN winter schedule) are
# correctly labeled at their actual final date, not at a calendar-year boundary.
print("Building tournament-end-date map...")
_TOURNAMENT_LABELS = {
    'FIFA World Cup':            ('World Cup Final',      1),
    'UEFA Euro':                 ('Euro Final',           2),
    'Copa América':              ('Copa América Final',   3),
    'AFC Asian Cup':             ('Asian Cup Final',      4),
    'African Cup of Nations':    ('African Cup Final',    5),
    'Gold Cup':                  ('Gold Cup Final',       6),
}
_games_full = pd.read_csv('all_soccer_games.csv', parse_dates=['date'])
_games_full = _games_full.dropna(subset=['home_score', 'away_score']).copy()
date_label_map = {}  # date_str -> (label_str, prestige)
for _tname, (_lbl, _prestige) in _TOURNAMENT_LABELS.items():
    _sub = _games_full[_games_full['tournament'] == _tname].sort_values('date')
    if _sub.empty:
        continue
    # An edition ends at a game with no further games within 90 days.
    _next_date = _sub['date'].shift(-1)
    _is_final = _next_date.isna() | ((_next_date - _sub['date']).dt.days > 90)
    for _d in _sub.loc[_is_final, 'date'].dt.date:
        _date_str = str(_d)
        if _date_str not in date_label_map or date_label_map[_date_str][1] > _prestige:
            date_label_map[_date_str] = (_lbl, _prestige)
print(f"  Found {len(date_label_map)} tournament-end snapshot dates")

print("Writing season standings files...")
all_seasons = sorted(df['year'].dropna().unique())

for season in all_seasons:
    season = int(season)
    sdf = df[df['year'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_date = str(rdf['date'].iloc[0])
        label = None
        prestige = None
        if snap_date in date_label_map:
            label, prestige = date_label_map[snap_date]
        teams_snap = []
        for _, r in rdf.iterrows():
            row = {
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
            era = display_name_at(r['country'], r['date'])
            if era and era != r['country']:
                row['display_name'] = era
            teams_snap.append(row)
        snapshots.append({'date': snap_date, 'label': label, 'prestige': prestige, 'teams': teams_snap})

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
            # Era-aware display: a 1980 Soviet Union podium row needs to
            # render "Soviet Union" inline while still linking to Russia.
            final_d = _tournament_final_date_by_yt.get((_tour, int(_yr)))
            era = display_name_at(team_name, final_d) if final_d else None
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
            if era and era != team_name:
                block['display_name'] = era
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
        # Note: keys use "Germany" not "West Germany" - the rating data layer
        # canonicalizes all eras to "Germany", so the seed must match for
        # post-1982 Germany entries to inherit the pre-1982 (West Germany) tally.
        'champion': {  # 1930-1978
            'Brazil': 3, 'Italy': 2, 'Germany': 2, 'Uruguay': 2,
            'Argentina': 1, 'England': 1,
        },
        'runner_up': {
            'Hungary': 2, 'Czechoslovakia': 2, 'Netherlands': 2,
            'Argentina': 1, 'Brazil': 1, 'Sweden': 1, 'Germany': 1, 'Italy': 1,
        },
        'third': {
            'Brazil': 2, 'Germany': 2,  # Germany 1934 + 1970
            'United States': 1, 'Sweden': 1, 'Austria': 1, 'France': 1,
            'Chile': 1, 'Portugal': 1, 'Poland': 1,
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


# Pre-rated tournament rows - displayed on the tournament tab for editions
# that pre-date our rated dataset. Each entry surfaces year + host + podium
# but has no rating/rank (data anchor is 1980). Title counts are computed
# below from running tallies within the pre-rated set so they line up with
# PRE_DATA_TOURNAMENT_COUNTS automatically.
PRE_RATED_PODIUMS = {
    'FIFA World Cup': [
        # oldest-first; will be inserted at the end of the newest-first entries list
        {'season': 1930, 'champion': 'Uruguay',   'runner_up': 'Argentina',      'third': 'United States'},
        {'season': 1934, 'champion': 'Italy',     'runner_up': 'Czechoslovakia', 'third': 'Germany'},
        {'season': 1938, 'champion': 'Italy',     'runner_up': 'Hungary',        'third': 'Brazil'},
        {'season': 1950, 'champion': 'Uruguay',   'runner_up': 'Brazil',         'third': 'Sweden'},
        {'season': 1954, 'champion': 'Germany',   'runner_up': 'Hungary',        'third': 'Austria'},
        {'season': 1958, 'champion': 'Brazil',    'runner_up': 'Sweden',         'third': 'France'},
        {'season': 1962, 'champion': 'Brazil',    'runner_up': 'Czechoslovakia', 'third': 'Chile'},
        {'season': 1966, 'champion': 'England',   'runner_up': 'Germany',        'third': 'Portugal'},
        {'season': 1970, 'champion': 'Brazil',    'runner_up': 'Italy',          'third': 'Germany'},
        {'season': 1974, 'champion': 'Germany',   'runner_up': 'Netherlands',    'third': 'Poland'},
        {'season': 1978, 'champion': 'Argentina', 'runner_up': 'Netherlands',    'third': 'Brazil'},
    ],
}


# Some tournaments in tournament_podiums.csv pre-date the rating data
# itself: games go back to 1980 but ratings only publish from 1986 onward
# (rolling window needs to fill first). For those editions, country_tournament_info()
# returns rating=None, rank=None, etc. Mark them pre_rated so the UI gives
# them the same gray-dash treatment as the hardcoded 1930-1978 rows, and
# strip the now-meaningless rating/rank/conf_rank fields from team blocks.
_first_rated_year = int(min(d.year for d in df['date'] if d is not None))
for tournament, entries in champions.items():
    for entry in entries:
        if entry['season'] >= _first_rated_year:
            continue
        entry['pre_rated'] = True
        for slot in ('champion', 'runner_up', 'third'):
            tb = entry.get(slot)
            if not tb:
                continue
            for k in ('rating', 'rank', 'conf_rank', 'confederation'):
                tb.pop(k, None)


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


# Append pre-rated podium rows (oldest editions of each tournament). These
# appear after the rated entries in the newest-first list, so they render at
# the bottom of the tournament tab. Cumulative title_counts are tallied
# within the pre-rated set so they end at the same totals as the seed dict
# above - the first rated entry then continues the count seamlessly.
def _pre_rated_block(team_name, count_key, count):
    # Mirror DILLON: omit rating/rank/conf_rank keys entirely so the UI's
    # pre-rated branch renders explicit gray dashes instead of via the
    # generic null-fallback (which is a slightly different shade).
    return {
        'team':    team_name,
        'flag':    country_flag(team_name),
        count_key: count,
    }

for tournament, pre_rows in PRE_RATED_PODIUMS.items():
    if tournament not in champions:
        champions[tournament] = []
    champ_counts = {}
    ru_counts    = {}
    third_counts = {}
    pre_entries_newest_first = []
    for row in pre_rows:  # oldest-first
        ct = row['champion']
        rt = row['runner_up']
        tt = row['third']
        champ_counts[ct] = champ_counts.get(ct, 0) + 1
        ru_counts[rt]    = ru_counts.get(rt, 0) + 1
        third_counts[tt] = third_counts.get(tt, 0) + 1
        pre_entries_newest_first.append({
            'season':     row['season'],
            'host_flags': host_flags(tournament, row['season']),
            'pre_rated':  True,   # entry-level flag, matches DILLON's PRE_RATED_SB_ROWS shape
            'champion':   _pre_rated_block(ct, 'title_count',      champ_counts[ct]),
            'runner_up':  _pre_rated_block(rt, 'runner_up_count',  ru_counts[rt]),
            'third':      _pre_rated_block(tt, 'third_count',      third_counts[tt]),
        })
    # Reverse to newest-first to match the rest of the entries list
    champions[tournament].extend(reversed(pre_entries_newest_first))


with open('docs/data/champions.json', 'w') as f:
    json.dump(champions, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
