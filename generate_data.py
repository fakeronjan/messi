"""
generate_data.py - reads messi_ratings_final.csv.gz and writes JSON for the MESSI web frontend.
Run after messi.py. Outputs to docs/data/.

Mirrors the ZIDANE site architecture (multiple competitions / leagues), adapted for
international soccer (countries instead of clubs, tournaments instead of league seasons).
"""

import pandas as pd
import json
import os
import re
import glob
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


_LM_RE = re.compile(r"^([WLDT]) (\d+-\d+) (vs\. \(N\) |vs\. |@ )(.+?)( \([^)]*\))?$")


def era_fix_lm(lm, d):
    """Rewrite a last_match string's opponent to its era-correct name (name-keyed)."""
    if not lm:
        return lm
    m = _LM_RE.match(lm)
    if not m:
        return lm
    res, score, venue, opp, comp = m.groups()
    era = display_name_at(opp, d)
    return f"{res} {score} {venue}{era}{comp or ''}" if era else lm


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
    'Zimbabwe': 'ZW', 'Zanzibar': '',  # Zanzibar - no flag emoji
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

# Defunct entities - no flag (intentional)
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
# Each entry: {tournament_short, finish} - used for honor badges with tournament context
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


# Per-(country, year) → World Cup record, SPLIT into group stage and knockout.
# Each team's FIRST 3 games of an edition are the group stage (true for every
# format 1986-2026+, including the 2026 expansion to a Round of 32); everything
# after is knockout. Group games use W-D-L (+points = 3W+D). Knockout has no
# draws: a game level after extra time goes into the separate pens W-L; games
# decided in regulation/ET go into the knockout W-L. Only the FIFA World Cup
# *finals* count - 'FIFA World Cup qualification' is a distinct label, excluded.
# Names in all_soccer_games.csv are canonical (West Germany folds into Germany).
#
# Opponent-strength annotation: each match line also carries the opponent's
# rank/rating as they stood going INTO that match (latest snapshot strictly
# before the match date), so the matches column reads e.g.
# "W 2-1 vs. France (#4, 2.70)".
from bisect import bisect_left, bisect_right
_team_snaps = {}
for _country, _sub in df[['country', 'date', 'rank', 'rating']].dropna(subset=['date']).groupby('country'):
    _sub = _sub.sort_values('date')
    _team_snaps[_country] = (list(_sub['date']), list(_sub['rank']), list(_sub['rating']))


def country_standing(country, match_date, inclusive=False):
    """A country's (rank, rating) relative to match_date; None if unknown.
    inclusive=False -> latest snapshot strictly BEFORE the date (going-in value);
    inclusive=True  -> latest snapshot ON or before the date (the game-day update,
    i.e. the post-match value once that day's result is baked in)."""
    snap = _team_snaps.get(country)
    if not snap:
        return None
    dates, ranks, ratings = snap
    i = bisect_right(dates, match_date) if inclusive else bisect_left(dates, match_date)
    if i == 0:
        return None
    rk, rt = ranks[i - 1], ratings[i - 1]
    if pd.isna(rk) or pd.isna(rt):
        return None
    return int(rk), round(float(rt), 2)


def opp_standing(opp, match_date):
    """Opponent (rank, rating) as of just before match_date; None if unknown."""
    return country_standing(opp, match_date, inclusive=False)


_wc_team_games = {}
for _, g in games[games['tournament'] == 'FIFA World Cup'].iterrows():
    if pd.isna(g['date']) or pd.isna(g['home_score']) or pd.isna(g['away_score']):
        continue
    yr = int(g['date'].year)
    hs, as_ = int(g['home_score']), int(g['away_score'])
    neutral = bool(g.get('neutral'))
    so = g.get('shootout_winner')
    so = so if isinstance(so, str) and so.strip() else None
    _wc_team_games.setdefault((g['home_team'], yr), []).append(
        {'date': g['date'], 'gf': hs, 'ga': as_, 'opp': g['away_team'],
         'home': True, 'neutral': neutral, 'won_so': so == g['home_team'], 'is_so': so is not None})
    _wc_team_games.setdefault((g['away_team'], yr), []).append(
        {'date': g['date'], 'gf': as_, 'ga': hs, 'opp': g['home_team'],
         'home': False, 'neutral': neutral, 'won_so': so == g['away_team'], 'is_so': so is not None})

# rec carries the split record plus 'matches': every WC game that edition as a
# {s, r, g} object - s is a "W 2-1 vs. (N) France"-style string (last_match
# format minus the competition suffix), r/g the opponent's pre-match rank and
# rating - in date order, for the per-edition match list cell. Knockout pens
# wins/losses are tracked separately (pens_w/pens_l) so the frontend can fold
# them into the knockout W-L and still annotate the shootouts.
_wc_record = {}
for (team, yr), gl in _wc_team_games.items():
    gl.sort(key=lambda m: m['date'])
    rec = {'g_w': 0, 'g_d': 0, 'g_l': 0, 'g_pts': 0, 'k_w': 0, 'k_l': 0, 'pens_w': 0, 'pens_l': 0}
    matches = []
    for i, m in enumerate(gl):
        gf, ga = m['gf'], m['ga']
        if gf > ga or (gf == ga and m['won_so']):
            letter = 'W'
        elif gf < ga or (gf == ga and m['is_so']):
            letter = 'L'
        else:
            letter = 'D'
        venue = ' vs. (N) ' if m['neutral'] else (' vs. ' if m['home'] else ' @ ')
        _st = opp_standing(m['opp'], m['date'])
        matches.append({'s': f"{letter} {gf}-{ga}{venue}{display_name_at(m['opp'], m['date']) or m['opp']}",
                        'r': _st[0] if _st else None,
                        'g': _st[1] if _st else None,
                        'd': f"{m['date'].month:02d}-{m['date'].day:02d}"})
        if i < 3:  # group stage: W-D-L on the scoreboard
            if gf > ga:
                rec['g_w'] += 1
            elif gf < ga:
                rec['g_l'] += 1
            else:
                rec['g_d'] += 1
        elif m['is_so']:  # knockout decided on penalties
            rec['pens_w' if m['won_so'] else 'pens_l'] += 1
        elif gf > ga:  # knockout decided in regulation/ET
            rec['k_w'] += 1
        elif gf < ga:
            rec['k_l'] += 1
    rec['g_pts'] = 3 * rec['g_w'] + rec['g_d']
    rec['matches'] = matches
    # The selected team's OWN rank/rating walk across the edition: N+1 boundary
    # standings in date order - index 0 is pre-tournament (strictly before the
    # first match), index i+1 is the post-match standing after game i. Mirrors
    # 'matches' so the frontend can render an offset stairstep beside it; the
    # final entry equals the end-of-tournament headline rating. {r, g} = rank,
    # rating (None,None when no snapshot exists, e.g. a country's WC debut).
    walk = []
    _pre = country_standing(team, gl[0]['date'], inclusive=False)
    walk.append({'r': _pre[0], 'g': _pre[1]} if _pre else {'r': None, 'g': None})
    for m in gl:
        _st = country_standing(team, m['date'], inclusive=True)
        walk.append({'r': _st[0], 'g': _st[1]} if _st else {'r': None, 'g': None})
    rec['team_walk'] = walk
    _wc_record[(team, yr)] = rec


def wc_record(country, year):
    if pd.isna(year):
        return None
    return _wc_record.get((country, int(year)))


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

# In-progress-year gate: the current calendar year is not over, so its
# fallback anchor must NOT read "End of year" (see in-progress-season-gate).
_CURRENT_YEAR = datetime.now(timezone.utc).year
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
    # 4. Fallback: last game-day of the year. A year still in progress (the
    #    current calendar year) is NOT over - label it "Current" so a live year
    #    (e.g. mid-2026 World Cup) isn't shown as a completed end-of-year snapshot.
    if chosen is None:
        chosen = (last_game, 'End of year' if year < _CURRENT_YEAR else 'Current')
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
            'rating_o':            round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
            'rating_d':            round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
            'rank_o':              int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
            'rank_d':              int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
            'games_played':        int(r['games_played']) if not pd.isna(r['games_played']) else 0,
            'last_match':          era_fix_lm(clean(r['last_match']), r['date']),
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
# moment they actually finished on the podium) - not whichever snapshot in
# the year had the highest rating. If a team medaled in multiple tournaments
# in one year, we use the rating from whichever is highest.
#
# Also requires games_played >= 6 in the rolling window - filters teams
# whose fakeronjan WLS rating comes from too small a sample of weak-confederation
# opponents (e.g. Tahiti 2012 OFC only had 5 OFC qualifier games).
print("Writing goat_teams.json...")
podiums = pd.read_csv('tournament_podiums.csv')
GOAT_MIN_GAMES = 6

# Medalist gate: 1st, 2nd, or 3rd in any major tournament. Matches the
# cross-site GOAT convention established 2026-05-11 - every other site
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
        'rating_o': snap.get('rating_o'),
        'rating_d': snap.get('rating_d'),
        'rank_o':   snap.get('rank_o'),
        'rank_d':   snap.get('rank_d'),
        'confederation': snap.get('confederation', ''),
        'games_played':  int(snap.get('games_played', 0)),
        'date':          final_date,
    })

print(f"  GOAT candidate_records: {len(candidate_records)} (from {len(_eligible_rows)} eligible podium rows)")

# Three GOAT views off the same medalist pool: overall rating, attack (rating_o),
# and defense (rating_d). Every row carries all three values so the frontend can
# show the offense/defense split and re-sort by metric; rank is positional to the
# active sort. rating_o + rating_d sums to rating (the split is re-anchored to it).
def _build_goat_table(records_df, sort_col):
    tbl = (records_df.sort_values(sort_col, ascending=False)
           .drop_duplicates(subset=['country', 'year'], keep='first')  # keep best if multi-tournament year
           .head(50)
           .reset_index(drop=True))
    rows = []
    for i, (_, r) in enumerate(tbl.iterrows()):
        entry = {
            'rank':                i + 1,
            'team':                r['country'],
            'flag':                country_flag(r['country']),
            'confederation':       clean(r['confederation']),
            'season':              int(r['year']),
            'rating':              round(float(r['rating']), 3),
            'rating_o':            round(float(r['rating_o']), 3) if not pd.isna(r['rating_o']) else None,
            'rating_d':            round(float(r['rating_d']), 3) if not pd.isna(r['rating_d']) else None,
            'rank_o':              int(r['rank_o']) if not pd.isna(r['rank_o']) else None,
            'rank_d':              int(r['rank_d']) if not pd.isna(r['rank_d']) else None,
            'tournament_finishes': country_year_finishes(r['country'], r['year']),
            'continental_winner':  1 if (r['country'], int(r['year'])) in _continental_winners else 0,
        }
        era = display_name_at(r['country'], r['date'])
        if era and era != r['country']:
            entry['display_name'] = era
        rows.append(entry)
    return rows

_goat_files = [('goat_teams.json', 'rating'), ('goat_teams_o.json', 'rating_o'), ('goat_teams_d.json', 'rating_d')]
if not candidate_records:
    # Defensive: empty candidates would crash sort_values. Emit empty files
    # and a diagnostic so we can investigate the underlying lookup failure.
    print("  WARNING: candidate_records is empty. Writing empty goat_teams*.json.")
    print(f"  Diagnostic: df shape={df.shape}, df.columns={list(df.columns)}")
    print(f"  Sample df['date'] dtypes: {df['date'].head(3).tolist()}")
    print(f"  Sample _tournament_final_date_map values: {list(_tournament_final_date_map.values())[:3]}")
    for _fname, _ in _goat_files:
        with open(f'docs/data/{_fname}', 'w') as f:
            json.dump([], f)
else:
    _records_df = pd.DataFrame(candidate_records)
    for _fname, _sort_col in _goat_files:
        payload = _build_goat_table(_records_df, _sort_col)
        with open(f'docs/data/{_fname}', 'w') as f:
            json.dump(payload, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
# Per team: keep only game days + EOS markers (avoids gigantic files)
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1) | (df['is_year_anchor'] == 1)].copy()
team_data = team_data.sort_values(['country', 'date'])

# (team, year) set of nations that actually played ≥1 game that year. Used to
# drop "ghost" year entries - without this, defunct nations (Czechoslovakia
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
            continue  # skip ghost year - team played 0 games
        finishes_for_year = country_year_finishes(team, season)
        won_continental = (team, int(season)) in _continental_winners
        _wc_rec_year = wc_record(team, season)
        rows = []
        for _, r in sdf.sort_values('date').iterrows():
            row = {
                'date':                str(r['date']),
                'rating':              round(float(r['rating']), 3) if not pd.isna(r['rating']) else None,
                'rank':                int(r['rank']) if not pd.isna(r['rank']) else None,
                'conf_rank':           int(r['conf_rank']) if not pd.isna(r['conf_rank']) else None,
                'rating_o':            round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
                'rating_d':            round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
                'rank_o':              int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
                'rank_d':              int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
                'last_match':          era_fix_lm(clean(r['last_match']), r['date']),
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
        # Attach the edition's World Cup record (the row the WC view filters on).
        # Completed editions: to the 'End of FIFA World Cup' anchor. In-progress
        # editions (live tournament, no anchor yet): to the LATEST snapshot, so
        # the row shows the current rating + record-so-far. The finish medal keys
        # off tournament_finishes, so an in-progress edition correctly shows none.
        if _wc_rec_year and rows:
            anchors = [row for row in rows if row['year_anchor_label'] == 'End of FIFA World Cup']
            if anchors:
                for row in anchors:
                    row['wc_record'] = _wc_rec_year
            else:
                rows[-1]['wc_record'] = _wc_rec_year
                rows[-1]['wc_in_progress'] = 1
        seasons[int(season)] = rows

    team_doc = {'team': team, 'flag': flag, 'confederation': confed, 'seasons': seasons}
    if hist_names:
        team_doc['historical_names'] = hist_names
    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump(team_doc, f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# Prune orphaned team files: when the data source renames a country (e.g.
# 'China PR' -> 'China') or drops one, the old-slug file lingers - unreachable
# from the UI (not in teams_index) but serving frozen, stale data. Remove any
# team file whose slug is no longer in the live index.
_live_team_files = {f"{t['slug']}.json" for t in teams_index}
for _f in glob.glob('docs/data/teams/*.json'):
    if os.path.basename(_f) not in _live_team_files:
        os.remove(_f)
        print(f"  Pruned orphaned team file: {os.path.basename(_f)}")

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
        # In-progress gate: the gap detector flags an edition's last PLAYED
        # game-day as its "final", but for a live tournament (no podium yet)
        # that's just the latest group game - don't label it "X Final". Only
        # label editions that have actually concluded (curated podium exists).
        if (_tname, _d.year) not in _podium_editions:
            continue
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
                'rating_o':            round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
                'rating_d':            round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
                'rank_o':              int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
                'rank_d':              int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
                'last_match':          era_fix_lm(clean(r['last_match']), r['date']),
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
# rating as of THAT specific tournament's final day - not whichever EOS row
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

# Pre-data tournament finishes - seed running counters with results from
# editions before our rated dataset begins (FIFA WC 1930-1978 not in our data,
# Euro 1960-1976, etc.). Keys use team names as they appear in our data.
# Defunct teams (Soviet Union, Yugoslavia, etc.) get seeded too - they just
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
        # Runner-ups and 3rd places not seeded - sparse pre-1987 records
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
    # Nations Leagues started in 2018/2019 - no pre-data history
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

# ── 6. World Cup win-probability simulation (Monte Carlo) ─────────────────────
# Runs only while a World Cup is in progress (group fixtures still unplayed).
# Engine: a Poisson goals model fit on every international match (going-in
# rating diff + home advantage) drives a Monte Carlo of the remaining group
# games + a rating-seeded knockout. The match model is well-calibrated on a
# 2022+ holdout (predicted P(win) tracks actual within ~2pts per bin); the
# knockout is rating-seeded (1 v N standard bracket) because the real FIFA
# bracket template + group letters are not in the feed.
import numpy as np
print("Simulating World Cup win probabilities...")

# Raw (unrounded) going-in rating from the per-team snapshot series. Match dates
# here are Timestamps (games['date'] was promoted above); _team_snaps keys are
# python dates, so normalize before the bisect.
def _raw_pre_rating(team, d):
    s = _team_snaps.get(team)
    if not s:
        return None
    dates, _ranks, ratings = s
    key = d.date() if hasattr(d, 'date') else d
    i = bisect_left(dates, key)
    return ratings[i - 1] if i > 0 else None

# Fit log λ = μ ± (a·rating_diff + b·home_adv): home + away stacked as 2N Poisson
# observations sharing (μ, a, b) with a sign flip, solved by Newton (no scipy).
_gm_rows = []
for _, _g in games.iterrows():
    if pd.isna(_g['home_score']) or pd.isna(_g['away_score']) or pd.isna(_g['date']):
        continue
    _rh = _raw_pre_rating(_g['home_team'], _g['date'])
    _ra = _raw_pre_rating(_g['away_team'], _g['date'])
    if _rh is None or _ra is None:
        continue
    _gm_rows.append((_rh - _ra, 0.0 if bool(_g.get('neutral')) else 1.0,
                     int(_g['home_score']), int(_g['away_score'])))
_gm = np.array(_gm_rows, float)
_X = np.vstack([np.column_stack([np.ones(len(_gm)),  _gm[:, 0],  _gm[:, 1]]),
                np.column_stack([np.ones(len(_gm)), -_gm[:, 0], -_gm[:, 1]])])
_yv = np.concatenate([_gm[:, 2], _gm[:, 3]])
_beta = np.zeros(3)
for _ in range(25):
    _lam = np.exp(_X @ _beta)
    _step = np.linalg.solve((_X * _lam[:, None]).T @ _X, _X.T @ (_yv - _lam))
    _beta += _step
    if np.abs(_step).max() < 1e-9:
        break
GM_MU, GM_A, GM_B = _beta
print(f"  goals model: mu={GM_MU:.3f} a={GM_A:.3f} b={GM_B:.3f} "
      f"({np.exp(GM_MU):.2f} goals/team, fit on {len(_gm):,} matches)")

# Phase-aware engine: simulates the group stage AND the knockout bracket, so it
# runs from kickoff to the final. During the group stage it reconstructs the
# groups + rating-seeds the knockout; once knockouts begin it rebuilds the live
# bracket from played games (honoring actual matchups + shootouts) and only
# rating-seeds the still-undetermined future pairings. Validated on the
# completed 2022 WC (deterministically crowns the real champion).
_wc_all = games[games['tournament'] == 'FIFA World Cup'].copy()
_wc_year = int(_wc_all['date'].dt.year.max()) if len(_wc_all) else None
wc_odds = {'edition': _wc_year,
           'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d'), 'teams': []}

def _ko_winner(row):
    """Knockout winner, honoring a penalty shootout on a regulation draw."""
    if row['home_score'] > row['away_score']: return row['home_team']
    if row['away_score'] > row['home_score']: return row['away_team']
    so = row.get('shootout_winner')
    return so if isinstance(so, str) and so.strip() else None

_ed = _wc_all[_wc_all['date'].dt.year == _wc_year].sort_values('date') if _wc_year else _wc_all.iloc[:0]
_nteams = pd.concat([_ed['home_team'], _ed['away_team']]).nunique() if len(_ed) else 0
if _nteams >= 4:
    _ngroups = _nteams // 4
    _grp = _ed.iloc[:_ngroups * 6]            # group stage precedes knockouts by date
    _ko = _ed.iloc[_ngroups * 6:]
    _grp_unplayed = _grp[_grp['home_score'].isna()]
    _grp_played = _grp[_grp['home_score'].notna()]
    _n_thirds = 8 if _ngroups == 12 else 0
    _cur = {_t: (_team_snaps[_t][2][-1] if _team_snaps.get(_t) and _team_snaps[_t][2] else 0.0)
            for _t in set(pd.concat([_ed['home_team'], _ed['away_team']]))}

    # Knockout games played so far -> winner, round, eliminations (skip 3rd place).
    _played_win = {}; _game_round = {}; _eliminated = set(); _cnt = {}
    for _, _g in _ko[_ko['home_score'].notna()].sort_values('date').iterrows():
        _a, _b = _g['home_team'], _g['away_team']
        if _a in _eliminated and _b in _eliminated:
            continue                          # 3rd-place playoff (both already lost)
        _w = _ko_winner(_g); _k = frozenset({_a, _b})
        _played_win[_k] = _w
        _game_round[_k] = max(_cnt.get(_a, 0), _cnt.get(_b, 0)) + 1
        _cnt[_a] = _cnt.get(_a, 0) + 1; _cnt[_b] = _cnt.get(_b, 0) + 1
        _eliminated.add(_b if _w == _a else _a)
    _ko_teams = set().union(*_played_win) if _played_win else set()

    # Groups = clean cliques from the group-stage games only.
    _adj = {}
    for _, _g in _grp.iterrows():
        _adj.setdefault(_g['home_team'], set()).add(_g['away_team'])
        _adj.setdefault(_g['away_team'], set()).add(_g['home_team'])
    _seen = set(); _groups = []
    for _t in _adj:
        if _t in _seen:
            continue
        _comp = set(); _st = [_t]
        while _st:
            _x = _st.pop()
            if _x in _comp:
                continue
            _comp.add(_x); _seen.add(_x); _st += [_y for _y in _adj[_x] if _y not in _comp]
        _groups.append(sorted(_comp))
    _gidx = {t: i for i, gp in enumerate(_groups) for t in gp}
    _teams = sorted(_adj)

    _rng = np.random.default_rng(20260101)    # fixed seed: odds move with data, not MC noise

    def _base_table():
        tb = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in _teams}
        for _, _g in _grp_played.iterrows():
            h, a, hs, as_ = _g['home_team'], _g['away_team'], int(_g['home_score']), int(_g['away_score'])
            tb[h]['gf'] += hs; tb[h]['gd'] += hs - as_; tb[a]['gf'] += as_; tb[a]['gd'] += as_ - hs
            if hs > as_: tb[h]['pts'] += 3
            elif hs < as_: tb[a]['pts'] += 3
            else: tb[h]['pts'] += 1; tb[a]['pts'] += 1
        return tb

    def _qualifiers(tb):
        win = []; ru = []; third = []
        for gp in _groups:
            rk = sorted(gp, key=lambda t: (tb[t]['pts'], tb[t]['gd'], tb[t]['gf'], _rng.random()), reverse=True)
            win.append(rk[0]); ru.append(rk[1])
            if len(rk) > 2: third.append(rk[2])
        third = sorted(third, key=lambda t: (tb[t]['pts'], tb[t]['gd'], tb[t]['gf'], _rng.random()), reverse=True)
        q = win + ru + third[:_n_thirds]
        if _ko_teams:    # every actual knockout participant must be a qualifier
            q = list(_ko_teams) + [x for x in q if x not in _ko_teams]
            q = q[:len(win) + len(ru) + _n_thirds]
        return q

    _STAGE = {32: 'r32', 16: 'r16', 8: 'qf', 4: 'sf', 2: 'final', 1: 'champ'}
    _KEYS = ('r32', 'r16', 'qf', 'sf', 'final', 'champ')

    def _build_bracket(quals):
        """Leaf-ordered bracket consistent with played KO games; rating-seed the rest."""
        blocks = [{'order': [q], 'rep': q} for q in quals]
        rr = lambda b: _cur.get(b['rep'], -99)
        level = 1
        while len(blocks) > 1:
            paired = set(); newb = []
            for b in blocks:
                if id(b) in paired:
                    continue
                partner = None
                for b2 in blocks:
                    if b2 is b or id(b2) in paired:
                        continue
                    k = frozenset({b['rep'], b2['rep']})
                    if k in _played_win and _game_round.get(k) == level:
                        partner = b2; break
                if partner is not None:
                    paired.add(id(b)); paired.add(id(partner))
                    newb.append({'order': b['order'] + partner['order'],
                                 'rep': _played_win[frozenset({b['rep'], partner['rep']})]})
            rem = [b for b in blocks if id(b) not in paired]
            rem.sort(key=rr, reverse=True)
            n = len(rem)
            for k in range(n // 2):
                a, b = rem[k], rem[n - 1 - k]
                newb.append({'order': a['order'] + b['order'], 'rep': a['rep'] if rr(a) >= rr(b) else b['rep']})
            blocks = newb; level += 1
        return blocks[0]['order'] if blocks else []

    # ── Known knockout brackets ───────────────────────────────────────────────
    # The feed carries the real R32 matchups (as scheduled fixtures) but NOT the
    # bracket TREE: which R32 winner meets which in R16 -> ... -> final. FIFA fixes
    # that template before the knockouts, so we encode it per edition as the 16 R32
    # ties in bracket-leaf order (top of bracket to bottom). Adjacent pairs are the
    # R16 ties, pairs-of-pairs the QFs, and so on up the binary tree. Source: FIFA
    # official 2026 bracket (M73-88 -> 89-96 -> 97-100 -> 101-102 -> 104). When the
    # edition is listed and every team has qualified, this replaces the rating-
    # seeded fallback so the sim follows the ACTUAL path to the trophy; otherwise it
    # degrades to the 1-v-N seeding. Played KO results are still pinned afterward
    # (the _played_pos walk reads _order_fixed regardless of how it was built).
    WC_KNOWN_BRACKETS = {
        2026: [  # leaf order: M74,M77,M73,M75,M83,M84,M81,M82,M76,M78,M79,M80,M86,M88,M85,M87
            ('Germany', 'Paraguay'),       ('France', 'Sweden'),
            ('South Africa', 'Canada'),    ('Netherlands', 'Morocco'),
            ('Portugal', 'Croatia'),       ('Spain', 'Austria'),
            ('United States', 'Bosnia and Herzegovina'), ('Belgium', 'Senegal'),
            ('Brazil', 'Japan'),           ('Ivory Coast', 'Norway'),
            ('Mexico', 'Ecuador'),         ('England', 'DR Congo'),
            ('Argentina', 'Cape Verde'),   ('Australia', 'Egypt'),
            ('Switzerland', 'Algeria'),    ('Colombia', 'Ghana'),
        ],
    }

    # Official group-stage draw (rosters only), for the Results column's "group +
    # finish" tag (A1, H2, K3...). Positions are NOT read from here - they're
    # computed live from the feed standings - so this is only the A-L membership,
    # validated against the feed's own cliques before use; a mismatch (e.g. a team-
    # name alias drift) skips the labels rather than mislabeling.
    WC_GROUP_ROSTERS = {
        2026: {
            'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
            'B': ['Switzerland', 'Canada', 'Bosnia and Herzegovina', 'Qatar'],
            'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
            'D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
            'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
            'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
            'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
            'H': ['Spain', 'Cape Verde', 'Uruguay', 'Saudi Arabia'],
            'I': ['France', 'Norway', 'Senegal', 'Iraq'],
            'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
            'K': ['Colombia', 'Portugal', 'DR Congo', 'Uzbekistan'],
            'L': ['England', 'Croatia', 'Ghana', 'Panama'],
        },
    }

    def _known_bracket_order(year, present):
        """Flat 32-team leaf order for `year`'s real bracket, or None to fall back.
        Returns None (with a warning) unless every listed team is present, so a data
        hiccup or alias drift never silently simulates the wrong bracket."""
        ties = WC_KNOWN_BRACKETS.get(year)
        if not ties:
            return None
        leaves = [t for tie in ties for t in tie]
        missing = sorted(t for t in leaves if t not in present)
        if missing:
            print(f"  Known {year} bracket skipped: teams absent from data {missing} "
                  f"- falling back to rating-seeded bracket.")
            return None
        return leaves

    # ── Vectorized Monte Carlo ────────────────────────────────────────────────
    # numpy-batched equivalent of the per-sim loops below (kept in git history):
    # all NSIM sims advance together as array columns instead of a Python loop,
    # cutting the in-tournament cost from ~8 min to seconds and letting NSIM go to
    # 1e6 for a near-deterministic estimate. The MODEL is identical (same Poisson
    # fit, same group tiebreaks, same rating-seeded bracket, same shootout logit);
    # only the RNG draw order changes, so odds differ from the old loop solely by
    # Monte-Carlo noise (validated to agree within ~3 standard errors).
    _NSIM = 1_000_000
    _CHUNK = 200_000          # cap peak memory; reach accumulates across chunks
    _tlist = list(_teams)
    _tix = {t: j for j, t in enumerate(_tlist)}
    _NT = len(_tlist)
    _ratv = np.array([_cur[t] for t in _tlist], float)

    def _vec_ko(leaf, reach, played_pos):
        """Vectorized single-elim KO. leaf: (n, L) team-col indices. reach: dict
        stage -> (NT,) int64 counts (mutated). played_pos: {(round, pair): winner
        col} for deterministic already-played games (empty during group stage)."""
        cur = leaf
        reach[_STAGE[cur.shape[1]]] += np.bincount(cur.ravel(), minlength=_NT)
        rnd = 0
        while cur.shape[1] > 1:
            a = cur[:, 0::2]; b = cur[:, 1::2]                 # (n, L/2)
            d = _ratv[a] - _ratv[b]
            ga = _rng.poisson(np.exp(GM_MU + GM_A * d))        # neutral venue: ha=0
            gb = _rng.poisson(np.exp(GM_MU - GM_A * d))
            a_wins = np.where(ga == gb,
                              _rng.random(d.shape) < 1.0 / (1.0 + np.exp(-0.25 * d)),
                              ga > gb)
            cur = np.where(a_wins, a, b)
            for (pr, pj), wcol in played_pos.items():          # fix played results
                if pr == rnd:
                    cur[:, pj] = wcol
            reach[_STAGE[cur.shape[1]]] += np.bincount(cur.ravel(), minlength=_NT)
            rnd += 1

    _reach_arr = {k: np.zeros(_NT, np.int64) for k in _KEYS}
    _grpfin, _ko_score = {}, {}                  # Results-column data (filled once groups end)
    _ko_path = {}                               # team -> ordered knockout path (all rounds)
    _group_done = len(_grp_unplayed) == 0 and len(_grp_played) > 0
    if _group_done:
        # Prefer the real FIFA bracket tree once groups end; the 32 leaf teams ARE
        # the qualifiers (ground truth), so we skip the thirds-tiebreak guesswork.
        _known = _known_bracket_order(_wc_year, set(_teams))
        if _known is not None:
            _order_fixed = _known
            _quals_fixed = _known
            print(f"  Using real {_wc_year} bracket tree from FIFA template "
                  f"(actual R32 matchups, not rating-seeded).")
        else:
            _quals_fixed = _qualifiers(_base_table())   # qualifiers fixed once groups end
            _order_fixed = _build_bracket(_quals_fixed)
        # Walk the fixed bracket's deterministic prefix to map already-played games
        # to (round, pair) so the vectorized KO can pin their winners.
        _played_pos = {}
        _det = list(_order_fixed); _r = 0
        while len(_det) > 1:
            _nd = []
            for _j in range(0, len(_det), 2):
                _x, _y = _det[_j], _det[_j + 1]
                _k = frozenset({_x, _y}) if (_x is not None and _y is not None) else None
                if _k is not None and _k in _played_win:
                    _w = _played_win[_k]; _played_pos[(_r, _j // 2)] = _tix[_w]; _nd.append(_w)
                else:
                    _nd.append(None)
            _det = _nd; _r += 1
        _leaf0 = np.array([_tix[t] for t in _order_fixed], np.int32)
        _done = 0
        while _done < _NSIM:
            _n = min(_CHUNK, _NSIM - _done)
            _vec_ko(np.tile(_leaf0, (_n, 1)), _reach_arr, _played_pos)
            _done += _n
        _alive = [t for t in _quals_fixed if t not in _eliminated]
        _in_progress = len(_alive) > 1
        _games_left = max(0, len(_alive) - 1)
        _qual_set = set(_quals_fixed)
        # Results-column data: group finish (validated draw + live feed standings),
        # plus each qualifier's R32 opponent and played result.
        _grp_rec = {}                          # team -> [W, D, L] in the group stage
        for _, _g in _grp_played.iterrows():
            _h, _a = _g['home_team'], _g['away_team']
            _hs, _gas = int(_g['home_score']), int(_g['away_score'])
            _grp_rec.setdefault(_h, [0, 0, 0]); _grp_rec.setdefault(_a, [0, 0, 0])
            if _hs > _gas: _grp_rec[_h][0] += 1; _grp_rec[_a][2] += 1
            elif _hs < _gas: _grp_rec[_h][2] += 1; _grp_rec[_a][0] += 1
            else: _grp_rec[_h][1] += 1; _grp_rec[_a][1] += 1
        _rosters = WC_GROUP_ROSTERS.get(_wc_year, {})
        _clique_sets = {frozenset(gp) for gp in _groups}
        if _rosters and len(_rosters) == len(_groups) and all(
                frozenset(ts) in _clique_sets for ts in _rosters.values()):
            _tb_fin = _base_table()
            for _L, _ts in _rosters.items():
                for _p, _t in enumerate(sorted(_ts, key=lambda t: (
                        _tb_fin[t]['pts'], _tb_fin[t]['gd'], _tb_fin[t]['gf']), reverse=True), 1):
                    _grpfin[_t] = f"{_L}{_p}"
        elif _rosters:
            print(f"  {_wc_year} group rosters don't match feed cliques - no group labels.")
        for _, _g in _ko[_ko['home_score'].notna()].iterrows():   # played KO scores
            _ko_score[frozenset({_g['home_team'], _g['away_team']})] = (
                _g['home_team'], int(_g['home_score']), int(_g['away_score']),
                _g['shootout_winner'] if isinstance(_g.get('shootout_winner'), str)
                and _g['shootout_winner'].strip() else None)
        # Each team's full knockout PATH: every round played (opponent + score +
        # W/L), plus the immediate pending matchup once both sides are known.
        # Walked off the fixed bracket tree, generalizing the old R32-only logic.
        _KO_ROUND_LABELS = ['R32', 'R16', 'QF', 'SF', 'Final']
        _walk = list(_order_fixed); _ri = 0
        while len(_walk) > 1:
            _lbl = _KO_ROUND_LABELS[_ri] if _ri < len(_KO_ROUND_LABELS) else 'R%d' % len(_walk)
            _next = []
            for _j in range(0, len(_walk), 2):
                _x, _y = _walk[_j], _walk[_j + 1]
                _k = frozenset({_x, _y}) if (_x is not None and _y is not None) else None
                _sc = _ko_score.get(_k) if _k is not None else None
                if _sc is not None:                       # played: record for both sides
                    _ht, _hs, _as, _so = _sc
                    for _tt, _opp in ((_x, _y), (_y, _x)):
                        _gf, _ga = (_hs, _as) if _ht == _tt else (_as, _hs)
                        _step = {'round': _lbl, 'opp': _opp, 'opp_flag': country_flag(_opp),
                                 'gf': _gf, 'ga': _ga,
                                 'won': (_gf > _ga) or (_gf == _ga and _so == _tt)}
                        if _so:
                            _step['pens'] = True
                        _ko_path.setdefault(_tt, []).append(_step)
                    _next.append(_played_win.get(_k))
                else:                                     # not yet played
                    if _k is not None:                    # both sides known: pending matchup
                        for _tt, _opp in ((_x, _y), (_y, _x)):
                            _ko_path.setdefault(_tt, []).append(
                                {'round': _lbl, 'opp': _opp,
                                 'opp_flag': country_flag(_opp), 'pending': True})
                    _next.append(None)
            _walk = _next; _ri += 1
    else:
        # Group stage: vectorize group sim -> qualifiers -> rating-seed -> KO.
        _bt = _base_table()
        _base_pts = np.array([_bt[t]['pts'] for t in _tlist], np.int32)
        _base_gd  = np.array([_bt[t]['gd']  for t in _tlist], np.int32)
        _base_gf  = np.array([_bt[t]['gf']  for t in _tlist], np.int32)
        _uh  = np.array([_tix[t] for t in _grp_unplayed['home_team']], int)
        _ua  = np.array([_tix[t] for t in _grp_unplayed['away_team']], int)
        _uha = np.array([0.0 if bool(x) else 1.0 for x in _grp_unplayed['neutral']])
        _ud  = _ratv[_uh] - _ratv[_ua]
        _ulam_h = np.exp(GM_MU + GM_A * _ud + GM_B * _uha)
        _ulam_a = np.exp(GM_MU - GM_A * _ud - GM_B * _uha)
        _ng = len(_grp_unplayed)
        _grp_cols = [np.array([_tix[t] for t in gp]) for gp in _groups]
        # rank -> bracket-leaf permutation, derived from _build_bracket so it
        # matches the original seeding exactly (played dicts are empty mid-group).
        _nq = len(_groups) * 2 + _n_thirds
        assert _nq & (_nq - 1) == 0, f"vectorized KO needs power-of-2 bracket, got {_nq}"
        _sample = sorted(_tlist, key=lambda t: _cur[t], reverse=True)[:_nq]
        _seed_rank = {t: i for i, t in enumerate(_sample)}    # _sample already rating-desc
        _P_perm = np.array([_seed_rank[t] for t in _build_bracket(_sample)])
        assert sorted(_P_perm.tolist()) == list(range(_nq))
        _ngrp = len(_groups)
        _done = 0
        while _done < _NSIM:
            _n = min(_CHUNK, _NSIM - _done)
            _pts = np.tile(_base_pts, (_n, 1))
            _gd  = np.tile(_base_gd,  (_n, 1))
            _gf  = np.tile(_base_gf,  (_n, 1))
            for _g in range(_ng):
                _gh = _rng.poisson(_ulam_h[_g], _n)
                _ga = _rng.poisson(_ulam_a[_g], _n)
                _h, _a = _uh[_g], _ua[_g]
                _gf[:, _h] += _gh; _gd[:, _h] += _gh - _ga
                _gf[:, _a] += _ga; _gd[:, _a] += _ga - _gh
                _pts[:, _h] += np.where(_gh > _ga, 3, np.where(_gh == _ga, 1, 0))
                _pts[:, _a] += np.where(_ga > _gh, 3, np.where(_gh == _ga, 1, 0))
            _wins = np.empty((_n, _ngrp), np.int32)
            _rus  = np.empty((_n, _ngrp), np.int32)
            _thtm = np.empty((_n, _ngrp), np.int32)
            _thky = np.empty((_n, _ngrp), np.float64)
            _rows = np.arange(_n)
            for _gi, _cols in enumerate(_grp_cols):
                _P = _pts[:, _cols].astype(np.float64)
                _G = _gd[:,  _cols].astype(np.float64)
                _F = _gf[:,  _cols].astype(np.float64)
                # composite sort key: pts >> gd >> gf >> uniform tiebreak (matches
                # the original sorted(..., (pts, gd, gf, random()), reverse=True)).
                _key = _P * 1e6 + _G * 1e3 + _F * 1e1 + _rng.random((_n, _cols.size))
                _ordr = np.argsort(-_key, axis=1)
                _wins[:, _gi] = _cols[_ordr[:, 0]]
                _rus[:,  _gi] = _cols[_ordr[:, 1]]
                _tp = _ordr[:, 2]
                _thtm[:, _gi] = _cols[_tp]
                _thky[:, _gi] = (_P[_rows, _tp] * 1e6 + _G[_rows, _tp] * 1e3
                                 + _F[_rows, _tp] * 1e1 + _rng.random(_n))  # fresh tiebreak
            _thord = np.argsort(-_thky, axis=1)
            _bestth = np.take_along_axis(_thtm, _thord[:, :_n_thirds], axis=1)
            _quals = np.concatenate([_wins, _rus, _bestth], axis=1)        # (n, nq)
            _seed_order = np.argsort(-_ratv[_quals], axis=1, kind='stable')  # rating-seed
            _seeds = np.take_along_axis(_quals, _seed_order, axis=1)
            _vec_ko(_seeds[:, _P_perm].astype(np.int32), _reach_arr, {})
            _done += _n
        _in_progress = True
        _games_left = len(_grp_unplayed)
        _qual_set = None

    _reach = {t: {k: int(_reach_arr[k][_tix[t]]) for k in _KEYS} for t in _teams}

    # Emit whenever the edition has teams - in progress OR complete. A finished
    # tournament keeps the tab alive as a permanent record (grid resolves to W/L
    # per round, champion on top); it only hides when there's no WC data at all.
    if _teams:
        for _t in sorted(_teams, key=lambda t: (_reach[t]['champ'], _reach[t]['final'], _reach[t]['sf']),
                         reverse=True):
            _r = _reach[_t]
            _elim = (_t in _eliminated) or (_qual_set is not None and _t not in _qual_set)
            _entry = {
                'team': _t, 'flag': country_flag(_t), 'rating': round(float(_cur[_t]), 2),
                'group': _gidx[_t], 'eliminated': bool(_elim),
                'champ': _r['champ'] / _NSIM, 'final': _r['final'] / _NSIM, 'sf': _r['sf'] / _NSIM,
                'qf': _r['qf'] / _NSIM, 'r16': _r['r16'] / _NSIM, 'r32': _r['r32'] / _NSIM,
            }
            if _t in _grpfin:
                _entry['grp'] = _grpfin[_t]
            if _t in _grp_rec:
                _entry['grp_rec'] = '%d-%d-%d' % tuple(_grp_rec[_t])
            if _t in _ko_path:
                _entry['ko_path'] = _ko_path[_t]
            wc_odds['teams'].append(_entry)
        wc_odds['n_sims'] = _NSIM
        wc_odds['games_left'] = _games_left
        wc_odds['phase'] = 'group' if not _group_done else 'knockout'
        wc_odds['complete'] = not _in_progress      # champion decided -> final results
        # The newest signal feeding this run: the most recent matchday's results,
        # so the tab can show what just happened and on which date.
        _done = _ed[_ed['home_score'].notna()]
        if len(_done):
            _last = _done['date'].max()
            wc_odds['latest_matchday'] = {
                'date': _last.strftime('%Y-%m-%d'),
                'games': [{'home': _g['home_team'], 'away': _g['away_team'],
                           'hg': int(_g['home_score']), 'ag': int(_g['away_score']),
                           'hflag': country_flag(_g['home_team']), 'aflag': country_flag(_g['away_team']),
                           'so': (_g['shootout_winner'] if isinstance(_g.get('shootout_winner'), str)
                                  and _g['shootout_winner'].strip() else None)}
                          for _, _g in _done[_done['date'] == _last].sort_values('home_team').iterrows()]
            }
        if _in_progress:
            print(f"  {_wc_year} WC ({wc_odds['phase']}): {_games_left} games to a champion, "
                  f"{_NSIM:,} sims. Favorite: {wc_odds['teams'][0]['team']} "
                  f"{wc_odds['teams'][0]['champ'] * 100:.1f}%")
        else:
            print(f"  {_wc_year} WC complete: {wc_odds['teams'][0]['team']} champions. "
                  f"Tab persists with final results.")
    else:
        print(f"  {_wc_year} WC: no teams; wrote empty odds (tab hides).")
else:
    print("  no World Cup data; wrote empty odds.")

with open('docs/data/wc_odds.json', 'w') as f:
    json.dump(wc_odds, f, separators=(',', ':'))


WC_HISTORY_TOURNAMENT = 'FIFA World Cup'   # MESSI's odds engine covers the WC finals


def _wc_history_snapshot(wc):
    """Self-contained per-game-state snapshot: the full team rows (so a historical
    date renders identically to live) plus the state metadata."""
    _ts = wc.get('teams', [])
    if not _ts:
        return None
    return {
        'date': wc.get('generated'),
        'phase': wc.get('phase', 'group') or 'group',
        'games_left': wc.get('games_left'),
        'complete': bool(wc.get('complete')),
        'teams': _ts,
    }


def _append_wc_history(wc, path='docs/data/wc_odds_history.json'):
    """Append a snapshot to the odds-history time-series. Namespaced by
    (tournament, edition) so a tournament/date selector can scrub any edition;
    de-duped within an edition by game-state (phase + games_left). Every played
    game changes games_left, so a re-run of the same state refreshes in place
    while each new result appends. Date is NOT a safe key - the full-R32 forecast
    and the first knockout result fall on the same calendar day.

    Only KNOCKOUT-phase snapshots are recorded: the schedule-aware odds follow
    the real FIFA bracket (commit 803223b onward). Group-stage odds rating-seed
    the knockout - a since-superseded model - so they're excluded from the
    history and the date selector. (For 2026 this is the 6/28 cutoff; it
    generalizes - every edition's group stage is auto-excluded.)"""
    _snap = _wc_history_snapshot(wc)
    if _snap is None or _snap['phase'] != 'knockout':
        return
    try:
        with open(path) as f:
            _hist = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _hist = {'tournaments': []}
    _ed = wc.get('edition')
    _buckets = _hist.setdefault('tournaments', [])
    _bucket = next((_b for _b in _buckets
                    if _b.get('tournament') == WC_HISTORY_TOURNAMENT and _b.get('edition') == _ed), None)
    if _bucket is None:
        _bucket = {'tournament': WC_HISTORY_TOURNAMENT, 'edition': _ed, 'snapshots': []}
        _buckets.append(_bucket)
    _snaps = _bucket.setdefault('snapshots', [])
    _key = (_snap['phase'], _snap['games_left'])
    for _i, _s in enumerate(_snaps):
        if (_s.get('phase'), _s.get('games_left')) == _key:
            _snaps[_i] = _snap
            break
    else:
        _snaps.append(_snap)
    with open(path, 'w') as f:
        json.dump(_hist, f, separators=(',', ':'))
    print(f"  WC odds history: {WC_HISTORY_TOURNAMENT} {_ed} -> {len(_snaps)} snapshots "
          f"(latest: {_snap['phase']} games_left={_snap['games_left']}).")


_append_wc_history(wc_odds)

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
