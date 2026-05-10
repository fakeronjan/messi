# ============================================================
# MESSI - International Soccer Power Rankings
# Massey Elo Soccer Strength Index
# Based on LOGAN (NBA) and OLANDIS (NFL) architecture
# ============================================================

import pandas as pd
import numpy as np
# rankit==0.2 uses deprecated numpy aliases (np.int, np.float, np.bool) removed in numpy 1.24+.
# Restore them before rankit import so the Massey solver works.
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'):  np.bool = bool
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')
import rankit
from rankit.Table import Table
from rankit.Ranker import MasseyRanker

# ============================================================
# PARAMETERS - adjust these as needed
# ============================================================

start_date         = '1980-01-01'   # first date to include in dataset
window_game_days   = 200            # rolling game-day window (matches ZIDANE architecture)
margin_cap         = 5              # max goal margin fed into Massey
shootout_margin    = 0.5            # margin assigned to a shootout win (AET level, shootout decides)
min_games          = 5              # minimum games in window for a team to appear in output
                                    # (15 was too strict for international play — Germany 2014 had 14 competitive
                                    #  games in window and got filtered. Mexico 1990 type artifacts are already
                                    #  filtered naturally by the Massey solver since they have no games in window.)
friendly_weight    = 0.75           # MESSI2 only: weight for friendlies vs competitive games
data_url           = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
shootouts_url      = 'https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv'

# ============================================================
# TOURNAMENT CONFIGURATION
# ============================================================

# Tournament importance weights — applied to margins in the Massey loop
# multiplied with recency weight so both factors compose naturally
TOURNAMENT_WEIGHTS = {
    # Tier 1 — 1.5x: FIFA World Cup finals tournament only
    'FIFA World Cup':                    1.5,
    'FIFA Confederations Cup':           1.5,
    'CONMEBOL–UEFA Cup of Champions':    1.5,

    # Tier 2 — 1.25x: Continental championships (finals tournaments only)
    'UEFA Euro':                         1.25,
    'Copa América':                      1.25,
    'African Cup of Nations':            1.25,
    'AFC Asian Cup':                     1.25,
    'Gold Cup':                          1.25,
    'CONCACAF Championship':             1.25,
    'Oceania Nations Cup':               1.25,
    'UEFA Nations League':               1.25,
    'CONCACAF Nations League':           1.25,
    'FIFA Series':                       1.25,
    'CONCACAF Series':                   1.25,
    'Superclásico de las Américas':      1.25,
}
# All other tournaments default to 1.0x
# These are knockout tournaments with a final + third-place match
PODIUM_TOURNAMENTS = [
    'FIFA World Cup',
    'UEFA Euro',
    'Copa América',
    'African Cup of Nations',     # source uses "African" (not "Africa")
    'AFC Asian Cup',
    'Gold Cup',                   # source name; no "CONCACAF" prefix
    'Oceania Nations Cup',        # source uses "Oceania" (not "OFC")
    'UEFA Nations League',
    'CONCACAF Nations League',
    'FIFA Confederations Cup',    # discontinued 2017 but kept for historical podiums
]

# Confederation lookup - maps each FIFA member to its confederation
# OFC = Oceania, AFC = Asia, CAF = Africa, UEFA = Europe,
# CONMEBOL = South America, CONCACAF = North/Central America & Caribbean
CONFEDERATION_MAP = {
    # UEFA
    'Albania': 'UEFA', 'Andorra': 'UEFA', 'Armenia': 'UEFA', 'Austria': 'UEFA',
    'Azerbaijan': 'UEFA', 'Belarus': 'UEFA', 'Belgium': 'UEFA', 'Bosnia and Herzegovina': 'UEFA',
    'Bulgaria': 'UEFA', 'Croatia': 'UEFA', 'Cyprus': 'UEFA', 'Czech Republic': 'UEFA',
    'Czechoslovakia': 'UEFA', 'Denmark': 'UEFA', 'England': 'UEFA', 'Estonia': 'UEFA',
    'Faroe Islands': 'UEFA', 'Finland': 'UEFA', 'France': 'UEFA', 'Georgia': 'UEFA',
    'Germany': 'UEFA', 'West Germany': 'UEFA', 'East Germany': 'UEFA', 'Gibraltar': 'UEFA',
    'Greece': 'UEFA', 'Hungary': 'UEFA', 'Iceland': 'UEFA', 'Israel': 'UEFA',
    'Italy': 'UEFA', 'Kazakhstan': 'UEFA', 'Kosovo': 'UEFA', 'Latvia': 'UEFA',
    'Liechtenstein': 'UEFA', 'Lithuania': 'UEFA', 'Luxembourg': 'UEFA', 'Malta': 'UEFA',
    'Moldova': 'UEFA', 'Monaco': 'UEFA', 'Montenegro': 'UEFA', 'Netherlands': 'UEFA',
    'North Macedonia': 'UEFA', 'Northern Ireland': 'UEFA', 'Norway': 'UEFA', 'Poland': 'UEFA',
    'Portugal': 'UEFA', 'Republic of Ireland': 'UEFA', 'Romania': 'UEFA', 'Russia': 'UEFA',
    'San Marino': 'UEFA', 'Scotland': 'UEFA', 'Serbia': 'UEFA', 'Slovakia': 'UEFA',
    'Slovenia': 'UEFA', 'Spain': 'UEFA', 'Sweden': 'UEFA', 'Switzerland': 'UEFA',
    'Turkey': 'UEFA', 'Ukraine': 'UEFA', 'Wales': 'UEFA', 'Yugoslavia': 'UEFA',
    'FR Yugoslavia': 'UEFA', 'Serbia and Montenegro': 'UEFA', 'Soviet Union': 'UEFA',

    # CONMEBOL
    'Argentina': 'CONMEBOL', 'Bolivia': 'CONMEBOL', 'Brazil': 'CONMEBOL',
    'Chile': 'CONMEBOL', 'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Paraguay': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',

    # CONCACAF
    'Anguilla': 'CONCACAF', 'Antigua and Barbuda': 'CONCACAF', 'Aruba': 'CONCACAF',
    'Bahamas': 'CONCACAF', 'Barbados': 'CONCACAF', 'Belize': 'CONCACAF',
    'Bermuda': 'CONCACAF', 'Bonaire': 'CONCACAF', 'British Virgin Islands': 'CONCACAF',
    'Canada': 'CONCACAF', 'Cayman Islands': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Cuba': 'CONCACAF', 'Curaçao': 'CONCACAF', 'Dominica': 'CONCACAF',
    'Dominican Republic': 'CONCACAF', 'El Salvador': 'CONCACAF', 'Grenada': 'CONCACAF',
    'Guatemala': 'CONCACAF', 'Guyana': 'CONCACAF', 'Haiti': 'CONCACAF',
    'Honduras': 'CONCACAF', 'Jamaica': 'CONCACAF', 'Mexico': 'CONCACAF',
    'Montserrat': 'CONCACAF', 'Nicaragua': 'CONCACAF', 'Panama': 'CONCACAF',
    'Puerto Rico': 'CONCACAF', 'Saint Kitts and Nevis': 'CONCACAF',
    'Saint Lucia': 'CONCACAF', 'Saint Martin': 'CONCACAF',
    'Saint Vincent and the Grenadines': 'CONCACAF', 'Sint Maarten': 'CONCACAF',
    'Suriname': 'CONCACAF', 'Trinidad and Tobago': 'CONCACAF',
    'Turks and Caicos Islands': 'CONCACAF', 'United States': 'CONCACAF',
    'US Virgin Islands': 'CONCACAF',

    # CAF
    'Algeria': 'CAF', 'Angola': 'CAF', 'Benin': 'CAF', 'Botswana': 'CAF',
    'Burkina Faso': 'CAF', 'Burundi': 'CAF', 'Cameroon': 'CAF', 'Cape Verde': 'CAF',
    'Central African Republic': 'CAF', 'Chad': 'CAF', 'Comoros': 'CAF',
    'DR Congo': 'CAF', 'Congo': 'CAF', 'Djibouti': 'CAF', 'Egypt': 'CAF',
    'Equatorial Guinea': 'CAF', 'Eritrea': 'CAF', 'Eswatini': 'CAF',
    'Ethiopia': 'CAF', 'Gabon': 'CAF', 'Gambia': 'CAF', 'Ghana': 'CAF',
    'Guinea': 'CAF', 'Guinea-Bissau': 'CAF', 'Ivory Coast': 'CAF', 'Kenya': 'CAF',
    'Lesotho': 'CAF', 'Liberia': 'CAF', 'Libya': 'CAF', 'Madagascar': 'CAF',
    'Malawi': 'CAF', 'Mali': 'CAF', 'Mauritania': 'CAF', 'Mauritius': 'CAF',
    'Morocco': 'CAF', 'Mozambique': 'CAF', 'Namibia': 'CAF', 'Niger': 'CAF',
    'Nigeria': 'CAF', 'Rwanda': 'CAF', 'São Tomé and Príncipe': 'CAF',
    'Senegal': 'CAF', 'Seychelles': 'CAF', 'Sierra Leone': 'CAF', 'Somalia': 'CAF',
    'South Africa': 'CAF', 'South Sudan': 'CAF', 'Sudan': 'CAF', 'Tanzania': 'CAF',
    'Togo': 'CAF', 'Tunisia': 'CAF', 'Uganda': 'CAF', 'Zambia': 'CAF',
    'Zimbabwe': 'CAF',

    # AFC
    'Afghanistan': 'AFC', 'Australia': 'AFC', 'Bahrain': 'AFC', 'Bangladesh': 'AFC',
    'Bhutan': 'AFC', 'Brunei': 'AFC', 'Cambodia': 'AFC', 'China': 'AFC',
    'China PR': 'AFC', 'Chinese Taipei': 'AFC', 'Guam': 'AFC', 'Hong Kong': 'AFC',
    'India': 'AFC', 'Indonesia': 'AFC', 'Iran': 'AFC', 'Iraq': 'AFC',
    'Japan': 'AFC', 'Jordan': 'AFC', 'Kuwait': 'AFC', 'Kyrgyzstan': 'AFC',
    'Laos': 'AFC', 'Lebanon': 'AFC', 'Macau': 'AFC', 'Malaysia': 'AFC',
    'Maldives': 'AFC', 'Mongolia': 'AFC', 'Myanmar': 'AFC', 'Nepal': 'AFC',
    'North Korea': 'AFC', 'Northern Mariana Islands': 'AFC', 'Oman': 'AFC',
    'Pakistan': 'AFC', 'Palestine': 'AFC', 'Philippines': 'AFC', 'Qatar': 'AFC',
    'Saudi Arabia': 'AFC', 'Singapore': 'AFC', 'South Korea': 'AFC',
    'Sri Lanka': 'AFC', 'Syria': 'AFC', 'Tajikistan': 'AFC', 'Thailand': 'AFC',
    'Timor-Leste': 'AFC', 'Turkmenistan': 'AFC', 'United Arab Emirates': 'AFC',
    'Uzbekistan': 'AFC', 'Vietnam': 'AFC', 'Yemen': 'AFC',

    # OFC
    'American Samoa': 'OFC', 'Cook Islands': 'OFC', 'Fiji': 'OFC',
    'New Caledonia': 'OFC', 'New Zealand': 'OFC', 'Papua New Guinea': 'OFC',
    'Samoa': 'OFC', 'Solomon Islands': 'OFC', 'Tahiti': 'OFC', 'Tonga': 'OFC',
    'Tuvalu': 'OFC', 'Vanuatu': 'OFC',
}

# ============================================================
# STEP 1 - LOAD AND CLEAN THE RAW DATA
# ============================================================

print("Loading results from GitHub...")
raw_df = pd.read_csv(data_url)
shootouts_df = pd.read_csv(shootouts_url)

print(f"Raw results loaded: {len(raw_df)} rows")
print(f"Shootouts loaded: {len(shootouts_df)} rows")

# Parse dates and filter to start date, exclude friendlies
raw_df['date'] = pd.to_datetime(raw_df['date'])
shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])

# Explicit allowlist of legitimate competitive tournaments
# Friendlies and non-FIFA/invitational tournaments are excluded by omission
COMPETITIVE_TOURNAMENTS = {
    # FIFA global
    'FIFA World Cup',
    'FIFA World Cup qualification',
    'FIFA Confederations Cup',
    'FIFA Series',
    'CONMEBOL–UEFA Cup of Champions',

    # UEFA
    'UEFA Euro',
    'UEFA Euro qualification',
    'UEFA Nations League',

    # CONMEBOL
    'Copa América',
    'Copa América qualification',
    'Superclásico de las Américas',

    # CONCACAF
    'Gold Cup',
    'Gold Cup qualification',
    'CONCACAF Championship',
    'CONCACAF Championship qualification',
    'CONCACAF Nations League',
    'CONCACAF Nations League qualification',
    'CONCACAF Series',
    'CFU Caribbean Cup',
    'CFU Caribbean Cup qualification',
    'UNCAF Cup',

    # CAF
    'African Cup of Nations',
    'African Cup of Nations qualification',
    'CECAFA Cup',
    'COSAFA Cup',
    'COSAFA Cup qualification',
    'WAFF Championship',
    'Arab Cup',
    'Arab Cup qualification',
    'Amílcar Cabral Cup',
    'UDEAC Cup',
    'West African Cup',

    # AFC
    'AFC Asian Cup',
    'AFC Asian Cup qualification',
    'AFC Challenge Cup',
    'AFC Challenge Cup qualification',
    'AFF Championship',
    'AFF Championship qualification',
    'ASEAN Championship',
    'ASEAN Championship qualification',
    'EAFF Championship',
    'EAFF Championship qualification',
    'Gulf Cup',
    'SAFF Cup',
    'WAFF Championship',
    'CAFA Nations Cup',

    # OFC
    'Oceania Nations Cup',
    'Oceania Nations Cup qualification',
    'MSG Prime Minister\'s Cup',
    'Melanesia Cup',
}

df = raw_df[
    (raw_df['date'] >= pd.to_datetime(start_date)) &
    (raw_df['tournament'].isin(COMPETITIVE_TOURNAMENTS))
].copy()

print(f"After date filter and friendly exclusion: {len(df)} rows")

# ============================================================
# STEP 2 - HANDLE SHOOTOUTS
# ============================================================
# The results.csv score reflects full time + extra time (AET).
# When a game goes to penalties, the AET score can be level (e.g. 0-0, 1-1)
# or occasionally a team leads after ET without needing penalties.
# We join shootouts to identify genuine shootout deciders and assign +0.5 margin.

# Normalize shootouts for joining
shootouts_df = shootouts_df[['date', 'home_team', 'away_team', 'winner']].copy()
shootouts_df.columns = ['date', 'home_team', 'away_team', 'shootout_winner']

df = pd.merge(df, shootouts_df, on=['date', 'home_team', 'away_team'], how='left')

# ============================================================
# STEP 3 - CALCULATE MARGINS
# ============================================================

df['raw_margin_home'] = df['home_score'] - df['away_score']
df['raw_margin_away'] = -df['raw_margin_home']

# For shootout games where AET score is level, assign +/- 0.5 to the winner
shootout_mask = (
    df['shootout_winner'].notna() &
    (df['raw_margin_home'] == 0)
)

df['margin_home'] = df['raw_margin_home']
df['margin_away'] = df['raw_margin_away']

df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'margin_home'] =  shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'margin_away'] = -shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'margin_home'] = -shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'margin_away'] =  shootout_margin

# Apply margin cap
df['margin_home'] = df['margin_home'].clip(-margin_cap, margin_cap)
df['margin_away'] = -df['margin_home']

# ============================================================
# STEP 4 - HOME FIELD ADJUSTMENT
# ============================================================
# neutral == TRUE means no home field advantage
# When not neutral, home team gets a -0.5 adjustment (nerf),
# away team gets +0.5 (buff), matching OLANDIS/LOGAN logic

home_field_advantage = 0.5

df['hfa'] = np.where(df['neutral'] == True, 0, home_field_advantage)
df['adj_margin_home'] = df['margin_home'] - df['hfa']
df['adj_margin_away'] = -df['adj_margin_home']

# ============================================================
# STEP 4b - TOURNAMENT WEIGHTS
# ============================================================
# REMOVED: tournament weights (WC 1.5x, continental 1.25x) created persistence
# distortion — a 4-year window means tournament-weighted games dominated team
# ratings long after the tournament ended (e.g., Mexico 1990 inflated by
# 1986 host-nation WC games carrying 1.5x weight). MESSI now matches ZIDANE's
# architecture: all competitive games at weight 1.0, let the network speak.

df['tournament_weight'] = 1.0

# ============================================================
# STEP 5 - WINS FOR STANDINGS
# ============================================================

df['home_win'] = np.where(
    df['margin_home'] > 0, 1,
    np.where(df['margin_home'] == 0, 0.5, 0)
)
df['away_win'] = 1 - df['home_win']

# Shootout win counts as a win in standings
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'home_win'] = 1
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'away_win'] = 0
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'home_win'] = 0
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'away_win'] = 1

# ============================================================
# STEP 6 - CONFEDERATION TAGS
# ============================================================

df['home_confederation'] = df['home_team'].map(CONFEDERATION_MAP).fillna('Unknown')
df['away_confederation'] = df['away_team'].map(CONFEDERATION_MAP).fillna('Unknown')

# Flag any teams we couldn't map so we can fix the lookup table
unknown_teams = set(
    df.loc[df['home_confederation'] == 'Unknown', 'home_team'].tolist() +
    df.loc[df['away_confederation'] == 'Unknown', 'away_team'].tolist()
)
if unknown_teams:
    print(f"WARNING: {len(unknown_teams)} teams not found in confederation map:")
    for t in sorted(unknown_teams):
        print(f"  - {t}")

# ============================================================
# STEP 7 - DATE IDs FOR ROLLING WINDOW
# ============================================================

df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Assign a grouped date ID (like LOGAN's grouped_date_id)
df['grouped_date_id'] = df.groupby('date').ngroup() + 1
df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

# Save the cleaned master game file
df.to_csv('all_soccer_games.csv', index=False)
print("Master game CSV saved: all_soccer_games.csv")

# ============================================================
# BUILD LAST MATCH STRINGS
# ============================================================
# Format: "W vs. France 2-1 (UEFA Euro)" or "L @ Brazil 0-3 (WCQ)"
# One row per team per game date — used later in the final merge.

df['home_score_int'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score_int'] = pd.to_numeric(df['away_score'], errors='coerce')
df = df.dropna(subset=['home_score_int', 'away_score_int']).copy()
df['home_score_int'] = df['home_score_int'].astype(int)
df['away_score_int'] = df['away_score_int'].astype(int)

# Result label from each team's perspective
df['home_result_flag'] = np.where(
    df['home_win'] == 1, 'W',
    np.where(df['home_win'] == 0.5, 'D', 'L')
)
df['away_result_flag'] = np.where(
    df['away_win'] == 1, 'W',
    np.where(df['away_win'] == 0.5, 'D', 'L')
)

# Venue label
df['home_venue'] = np.where(df['neutral'] == True, ' vs. (N) ', ' vs. ')
df['away_venue'] = np.where(df['neutral'] == True, ' vs. (N) ', ' @ ')

# Full last match string for the home team: e.g. "W vs. Germany 2-1 (FIFA World Cup)"
df['home_last_match'] = (
    df['home_result_flag'] +
    df['home_venue'] +
    df['away_team'] + ' ' +
    df['home_score_int'].map(str) + '-' + df['away_score_int'].map(str) +
    ' (' + df['tournament'] + ')'
)

# Full last match string for the away team: e.g. "L @ Germany 1-2 (FIFA World Cup)"
df['away_last_match'] = (
    df['away_result_flag'] +
    df['away_venue'] +
    df['home_team'] + ' ' +
    df['away_score_int'].map(str) + '-' + df['home_score_int'].map(str) +
    ' (' + df['tournament'] + ')'
)

# Build a lookup: (date, team) -> last_match string. Season included so the
# downstream merge_asof can scope carry-forward to within the team's same
# season (calendar year for MESSI) — at the start of a new year, teams that
# haven't played yet correctly show empty rather than their last result from
# the prior year.
lastmatch_home = df[['date', 'home_team', 'home_last_match']].copy()
lastmatch_home.columns = ['date', 'name', 'last_match']
lastmatch_away = df[['date', 'away_team', 'away_last_match']].copy()
lastmatch_away.columns = ['date', 'name', 'last_match']
lastmatch_df = pd.concat([lastmatch_home, lastmatch_away], axis=0).reset_index(drop=True)
lastmatch_df['date'] = pd.to_datetime(lastmatch_df['date']).dt.date
lastmatch_df['season'] = pd.to_datetime(lastmatch_df['date']).apply(lambda d: d.year)

# ============================================================
# STEP 8 - ROLLING MASSEY RATINGS (MESSI RATINGS)
# ============================================================
# One snapshot per game-day, rolling 4-year window with linear recency weighting.
# Incremental: only computes dates not already in messi_ratings.csv

print("Starting MESSI rating calculations...")

max_date_id = int(df['grouped_date_id'].max())
min_date_id = 1  # start from first date that has a full window's worth of history

# Load existing ratings if available, otherwise start fresh
try:
    messi_df = pd.read_csv('messi_ratings.csv')
    max_date_id_ranked = int(messi_df['ranking_id'].max())
    min_date_id_ranked = int(messi_df['ranking_id'].min())
    print(f"Existing ratings found. Ranked IDs: {min_date_id_ranked} to {max_date_id_ranked}")
except FileNotFoundError:
    messi_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_date_id_ranked = -1
    min_date_id_ranked = -1
    print("No existing ratings found - running full history from scratch.")

working_df = pd.DataFrame()

for i in range(min_date_id, max_date_id + 1):

    # Skip already-ranked dates (incremental logic from LOGAN)
    if min_date_id_ranked <= i <= max_date_id_ranked:
        continue

    # Slice the rolling game-day window (matches ZIDANE architecture)
    current_date = df.loc[df['grouped_date_id'] == i, 'date'].max()

    working_df = df.loc[
        (df['grouped_date_id'] >= i - window_game_days + 1) &
        (df['grouped_date_id'] <= i)
    ].copy()

    if len(working_df) < 10:
        # Not enough data to produce meaningful ratings yet
        continue

    # Linear recency weight by game-days: oldest in window → near 0, most recent → 1.0
    working_df['game_days_ago'] = i - working_df['grouped_date_id']
    working_df['date_weight'] = 1 - (working_df['game_days_ago'] / window_game_days)

    # Scale margins by recency weight only — no tournament weight (see STEP 4b)
    working_df['weighted_margin_home'] = (
        working_df['adj_margin_home'] *
        working_df['date_weight']
    )
    working_df['weighted_margin_away'] = -working_df['weighted_margin_home']
    # Drop zero-weighted rows to avoid Massey solver issues
    working_df = working_df[working_df['weighted_margin_home'] != 0]
    if len(working_df) < 10:
        continue

    season = current_date.year

    # Print progress only when the month changes
    current_ym = current_date.strftime('%Y-%m')
    if 'last_printed_ym' not in dir() or current_ym != last_printed_ym:
        pct = round(100 * i / max_date_id)
        print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct}% complete)")
        last_printed_ym = current_ym

    # Build Massey table and run rankings
    soccer_table = Table(
        working_df,
        ['home_team', 'away_team', 'weighted_margin_home', 'weighted_margin_away']
    )
    messi_massey = MasseyRanker(soccer_table)
    ranked = messi_massey.rank()

    ranked['ranking_id']   = i
    ranked['ranking_date'] = current_date.date()
    ranked['season']       = season

    # Games played in the rolling window for each team
    home_gp = working_df.groupby('home_team').size().reset_index(name='gp_home')
    away_gp = working_df.groupby('away_team').size().reset_index(name='gp_away')
    home_gp.columns = ['name', 'gp_home']
    away_gp.columns = ['name', 'gp_away']
    gp = pd.merge(home_gp, away_gp, on='name', how='outer').fillna(0)
    gp['games_played'] = (gp['gp_home'] + gp['gp_away']).astype(int)
    gp = gp[['name', 'games_played']]
    ranked = pd.merge(ranked, gp, on='name', how='left')
    ranked['games_played'] = ranked['games_played'].fillna(0).astype(int)

    messi_df = pd.concat([messi_df, ranked], axis=0, sort=False).reset_index(drop=True)

messi_df.sort_values(['ranking_id', 'name'], ascending=[True, True], inplace=True)
messi_df.drop_duplicates(subset=None, keep='first', inplace=True)
messi_df['ranking_date'] = pd.to_datetime(messi_df['ranking_date']).dt.date

messi_df.to_csv('messi_ratings.csv', index=False)
print("messi_ratings.csv saved!")

# ============================================================
# STEP 8b - BUILD MESSI2 DATASET (all non-U23 matches)
# ============================================================
# MESSI2 uses the same competitive tournament weights as MESSI1,
# but adds back all excluded matches (friendlies, invitationals etc.)
# at a base weight of 0.75x. U-23 tournaments remain excluded.

U23_TOURNAMENTS = {
    'Asian Games', 'Southeast Asian Games', 'South Asian Games',
    'East Asian Games', 'Pacific Games', 'South Pacific Games',
    'South Pacific Mini Games', 'Pacific Mini Games',
    'All-African Games', 'Afro-Asian Games', 'Inter Games',
    'Olympic Games',
}

df2 = raw_df[
    (raw_df['date'] >= pd.to_datetime(start_date)) &
    (~raw_df['tournament'].isin(U23_TOURNAMENTS))
].copy()

print(f"MESSI2 dataset: {len(df2)} rows after U23 exclusion")

# Apply same shootout, margin, HFA processing as MESSI1
df2 = pd.merge(df2, shootouts_df, on=['date', 'home_team', 'away_team'], how='left')

df2['raw_margin_home'] = df2['home_score'] - df2['away_score']
df2['raw_margin_away'] = -df2['raw_margin_home']

shootout_mask2 = (df2['shootout_winner'].notna() & (df2['raw_margin_home'] == 0))

df2['margin_home'] = df2['raw_margin_home']
df2['margin_away'] = df2['raw_margin_away']
df2.loc[shootout_mask2 & (df2['shootout_winner'] == df2['home_team']), 'margin_home'] =  shootout_margin
df2.loc[shootout_mask2 & (df2['shootout_winner'] == df2['home_team']), 'margin_away'] = -shootout_margin
df2.loc[shootout_mask2 & (df2['shootout_winner'] == df2['away_team']), 'margin_home'] = -shootout_margin
df2.loc[shootout_mask2 & (df2['shootout_winner'] == df2['away_team']), 'margin_away'] =  shootout_margin

df2['margin_home'] = df2['margin_home'].clip(-margin_cap, margin_cap)
df2['margin_away'] = -df2['margin_home']

df2['hfa'] = np.where(df2['neutral'] == True, 0, home_field_advantage)
df2['adj_margin_home'] = df2['margin_home'] - df2['hfa']
df2['adj_margin_away'] = -df2['adj_margin_home']

# MESSI2 weights: competitive games at 1.0x, friendlies/invitationals at friendly_weight (0.75x).
# All competitive tiers (WC, continental, qualifying) treated equally — let the network speak.
df2['tournament_weight'] = np.where(
    df2['tournament'].isin(COMPETITIVE_TOURNAMENTS),
    1.0,
    friendly_weight
)

df2['date'] = pd.to_datetime(df2['date'])
df2.sort_values('date', inplace=True)
df2.reset_index(drop=True, inplace=True)
df2['grouped_date_id'] = df2.groupby('date').ngroup() + 1

# ============================================================
# STEP 8c - ROLLING MASSEY RATINGS (MESSI2)
# ============================================================

print("Starting MESSI2 rating calculations...")

max_date_id2 = int(df2['grouped_date_id'].max())

try:
    messi2_df = pd.read_csv('messi2_ratings.csv.gz')
    max_date_id2_ranked = int(messi2_df['ranking_id'].max())
    min_date_id2_ranked = int(messi2_df['ranking_id'].min())
    print(f"Existing MESSI2 ratings found. Ranked IDs: {min_date_id2_ranked} to {max_date_id2_ranked}")
except FileNotFoundError:
    messi2_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_date_id2_ranked = -1
    min_date_id2_ranked = -1
    print("No existing MESSI2 ratings found - running full history from scratch.")

for i in range(min_date_id, max_date_id2 + 1):

    if min_date_id2_ranked <= i <= max_date_id2_ranked:
        continue

    current_date2 = df2.loc[df2['grouped_date_id'] == i, 'date'].max()
    if pd.isnull(current_date2) or len(df2.loc[df2['grouped_date_id'] == i]) == 0:
        continue

    working2_df = df2.loc[
        (df2['grouped_date_id'] >= i - window_game_days + 1) &
        (df2['grouped_date_id'] <= i)
    ].copy()

    if len(working2_df) < 10:
        continue

    working2_df['game_days_ago'] = i - working2_df['grouped_date_id']
    working2_df['date_weight']   = 1 - (working2_df['game_days_ago'] / window_game_days)

    working2_df['weighted_margin_home'] = (
        working2_df['adj_margin_home'] *
        working2_df['date_weight'] *
        working2_df['tournament_weight']
    )
    working2_df['weighted_margin_away'] = -working2_df['weighted_margin_home']

    # Drop games where weighted margin is exactly zero — causes Massey solver issues
    working2_df = working2_df[working2_df['weighted_margin_home'] != 0]

    if len(working2_df) < 10:
        continue

    season2 = current_date2.year

    current_ym2 = current_date2.strftime('%Y-%m')
    if 'last_printed_ym2' not in dir() or current_ym2 != last_printed_ym2:
        pct2 = round(100 * i / max_date_id2)
        print(f"  MESSI2: {current_date2.strftime('%B %Y')} ({pct2}% complete)")
        last_printed_ym2 = current_ym2

    try:
        soccer_table2 = Table(
            working2_df,
            ['home_team', 'away_team', 'weighted_margin_home', 'weighted_margin_away']
        )
        messi2_massey = MasseyRanker(soccer_table2)
        ranked2 = messi2_massey.rank()

        # Skip if rankit produced NaN/inf ratings
        if ranked2['rating'].isna().any() or np.isinf(ranked2['rating']).any():
            continue

        ranked2['ranking_id']   = i
        ranked2['ranking_date'] = current_date2.date()
        ranked2['season']       = season2

        # Games played in the rolling window for each team (used for min_games filter)
        home_gp2 = working2_df.groupby('home_team').size().reset_index(name='gp_home')
        away_gp2 = working2_df.groupby('away_team').size().reset_index(name='gp_away')
        home_gp2.columns = ['name', 'gp_home']
        away_gp2.columns = ['name', 'gp_away']
        gp2 = pd.merge(home_gp2, away_gp2, on='name', how='outer').fillna(0)
        gp2['games_played2'] = (gp2['gp_home'] + gp2['gp_away']).astype(int)
        ranked2 = pd.merge(ranked2, gp2[['name', 'games_played2']], on='name', how='left')
        ranked2['games_played2'] = ranked2['games_played2'].fillna(0).astype(int)

        messi2_df = pd.concat([messi2_df, ranked2], axis=0, sort=False).reset_index(drop=True)

    except Exception as e:
        print(f"  MESSI2: skipping date ID {i} due to solver error: {e}")
        continue

messi2_df.sort_values(['ranking_id', 'name'], ascending=[True, True], inplace=True)
messi2_df.drop_duplicates(subset=None, keep='first', inplace=True)
messi2_df['ranking_date'] = pd.to_datetime(messi2_df['ranking_date']).dt.date
messi2_df.rename(columns={'rating': 'rating2', 'rank': 'rank2'}, inplace=True)

messi2_df.to_csv('messi2_ratings.csv.gz', index=False)
print("messi2_ratings.csv.gz saved!")

# ============================================================
# STEP 8d - BUILD MESSI3 DATASET (FIFA World Cup ONLY — qualifying + finals)
# ============================================================
# MESSI3 isolates the highest-stakes tournament: just WC qualifying + WC finals.
# Sparser data per team but cleanest signal of "who actually performs at the World Cup level."

WC_TOURNAMENTS = {'FIFA World Cup', 'FIFA World Cup qualification'}

df3 = raw_df[
    (raw_df['date'] >= pd.to_datetime(start_date)) &
    (raw_df['tournament'].isin(WC_TOURNAMENTS))
].copy()

print(f"MESSI3 (WC only) dataset: {len(df3)} rows")

df3 = pd.merge(df3, shootouts_df, on=['date', 'home_team', 'away_team'], how='left')
df3['raw_margin_home'] = df3['home_score'] - df3['away_score']
df3['raw_margin_away'] = -df3['raw_margin_home']
shootout_mask3 = (df3['shootout_winner'].notna() & (df3['raw_margin_home'] == 0))
df3['margin_home'] = df3['raw_margin_home']
df3['margin_away'] = df3['raw_margin_away']
df3.loc[shootout_mask3 & (df3['shootout_winner'] == df3['home_team']), 'margin_home'] =  shootout_margin
df3.loc[shootout_mask3 & (df3['shootout_winner'] == df3['home_team']), 'margin_away'] = -shootout_margin
df3.loc[shootout_mask3 & (df3['shootout_winner'] == df3['away_team']), 'margin_home'] = -shootout_margin
df3.loc[shootout_mask3 & (df3['shootout_winner'] == df3['away_team']), 'margin_away'] =  shootout_margin

df3['margin_home'] = df3['margin_home'].clip(-margin_cap, margin_cap)
df3['margin_away'] = -df3['margin_home']
df3['hfa'] = np.where(df3['neutral'] == True, 0, home_field_advantage)
df3['adj_margin_home'] = df3['margin_home'] - df3['hfa']
df3['adj_margin_away'] = -df3['adj_margin_home']

df3['date'] = pd.to_datetime(df3['date'])
df3.sort_values('date', inplace=True)
df3.reset_index(drop=True, inplace=True)
df3['grouped_date_id'] = df3.groupby('date').ngroup() + 1

# ============================================================
# STEP 8e - ROLLING MASSEY RATINGS (MESSI3)
# ============================================================

print("Starting MESSI3 (WC-only) rating calculations...")

max_date_id3 = int(df3['grouped_date_id'].max())

try:
    messi3_df = pd.read_csv('messi3_ratings.csv')
    max_date_id3_ranked = int(messi3_df['ranking_id'].max())
    min_date_id3_ranked = int(messi3_df['ranking_id'].min())
    print(f"Existing MESSI3 ratings found. Ranked IDs: {min_date_id3_ranked} to {max_date_id3_ranked}")
except FileNotFoundError:
    messi3_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_date_id3_ranked = -1
    min_date_id3_ranked = -1
    print("No existing MESSI3 ratings found - running full history from scratch.")

last_printed_ym3 = None
for i in range(1, max_date_id3 + 1):

    if min_date_id3_ranked <= i <= max_date_id3_ranked:
        continue

    current_date3 = df3.loc[df3['grouped_date_id'] == i, 'date'].max()
    if pd.isnull(current_date3):
        continue

    working3_df = df3.loc[
        (df3['grouped_date_id'] >= i - window_game_days + 1) &
        (df3['grouped_date_id'] <= i)
    ].copy()

    if len(working3_df) < 10:
        continue

    working3_df['game_days_ago'] = i - working3_df['grouped_date_id']
    working3_df['date_weight']   = 1 - (working3_df['game_days_ago'] / window_game_days)
    working3_df['weighted_margin_home'] = working3_df['adj_margin_home'] * working3_df['date_weight']
    working3_df['weighted_margin_away'] = -working3_df['weighted_margin_home']

    working3_df = working3_df[working3_df['weighted_margin_home'] != 0]
    if len(working3_df) < 10:
        continue

    season3 = current_date3.year
    current_ym3 = current_date3.strftime('%Y-%m')
    if current_ym3 != last_printed_ym3:
        pct3 = round(100 * i / max_date_id3)
        print(f"  MESSI3: {current_date3.strftime('%B %Y')} ({pct3}% complete)")
        last_printed_ym3 = current_ym3

    try:
        soccer_table3 = Table(
            working3_df,
            ['home_team', 'away_team', 'weighted_margin_home', 'weighted_margin_away']
        )
        messi3_massey = MasseyRanker(soccer_table3)
        ranked3 = messi3_massey.rank()

        if ranked3['rating'].isna().any() or np.isinf(ranked3['rating']).any():
            continue

        ranked3['ranking_id']   = i
        ranked3['ranking_date'] = current_date3.date()
        ranked3['season']       = season3

        home_gp3 = working3_df.groupby('home_team').size().reset_index(name='gp_home')
        away_gp3 = working3_df.groupby('away_team').size().reset_index(name='gp_away')
        home_gp3.columns = ['name', 'gp_home']
        away_gp3.columns = ['name', 'gp_away']
        gp3 = pd.merge(home_gp3, away_gp3, on='name', how='outer').fillna(0)
        gp3['games_played3'] = (gp3['gp_home'] + gp3['gp_away']).astype(int)
        ranked3 = pd.merge(ranked3, gp3[['name', 'games_played3']], on='name', how='left')
        ranked3['games_played3'] = ranked3['games_played3'].fillna(0).astype(int)

        messi3_df = pd.concat([messi3_df, ranked3], axis=0, sort=False).reset_index(drop=True)

    except Exception as e:
        print(f"  MESSI3: skipping date ID {i} due to solver error: {e}")
        continue

messi3_df.sort_values(['ranking_id', 'name'], ascending=[True, True], inplace=True)
messi3_df.drop_duplicates(subset=None, keep='first', inplace=True)
messi3_df['ranking_date'] = pd.to_datetime(messi3_df['ranking_date']).dt.date
messi3_df.rename(columns={'rating': 'rating3', 'rank': 'rank3'}, inplace=True)

messi3_df.to_csv('messi3_ratings.csv', index=False)
print("messi3_ratings.csv saved!")

# ============================================================
# STEP 9 - TOURNAMENT PODIUM FLAGS
# ============================================================
# For each edition of every PODIUM_TOURNAMENT, identify 1st, 2nd, and 3rd place.
# Strategy:
#   - The final is the last match between two teams in that tournament edition.
#     The winner = 1st place, loser = 2nd place.
#   - The third-place match is the second-to-last match day's match between
#     the two teams who did NOT play in the final.
#   - We identify tournament editions by (tournament, year).

print("Calculating tournament podium flags...")

podium_df = df[df['tournament'].isin(PODIUM_TOURNAMENTS)].copy()
podium_df['year'] = podium_df['date'].dt.year

podium_records = []

for (tournament, year), group in podium_df.groupby(['tournament', 'year']):

    group = group.sort_values('date')
    last_date = group['date'].max()

    # Final = match(es) on the last date of the tournament
    final_games = group[group['date'] == last_date]

    if len(final_games) == 0:
        continue

    # Some tournaments play the FINAL and the 3rd-PLACE PLAYOFF on the same
    # day (Oceania Nations Cup, Africa Cup, sometimes others). Pick the
    # actual final by walking back the bracket: in the final both teams
    # WON their previous match (the semifinals); in the 3rd-place playoff
    # both teams LOST their previous match.
    def _won_last_pre_final(team):
        prev = group[
            (group['date'] < last_date) &
            ((group['home_team'] == team) | (group['away_team'] == team))
        ].sort_values('date')
        if prev.empty:
            return None
        last = prev.iloc[-1]
        if last['home_team'] == team:
            if last['margin_home'] > 0: return True
            if last['margin_home'] < 0: return False
        else:
            if last['margin_home'] < 0: return True
            if last['margin_home'] > 0: return False
        # Tied — check shootout
        sw = last.get('shootout_winner')
        if pd.notna(sw):
            return sw == team
        return None

    final = None
    for cand in reversed(list(final_games.itertuples(index=False))):
        cand_home_won = _won_last_pre_final(cand.home_team)
        cand_away_won = _won_last_pre_final(cand.away_team)
        if cand_home_won is True and cand_away_won is True:
            final = cand._asdict() if hasattr(cand, '_asdict') else dict(zip(final_games.columns, cand))
            break
    if final is None:
        # Bracket-walk failed (couldn't validate two semifinal winners). This
        # happens for in-progress tournaments where the data hasn't reached
        # the knockout stage yet. Only fall back to "last game on last date"
        # if the tournament is conclusively complete — at least 14 days past
        # its latest match. Otherwise skip and avoid awarding a fake medal.
        last_dt = last_date.date() if hasattr(last_date, 'date') else last_date
        if (date.today() - last_dt).days < 14:
            continue
        final = final_games.iloc[-1].to_dict()

    home = final['home_team']
    away = final['away_team']
    home_margin = final['margin_home']

    if home_margin > 0:
        champion   = home
        runner_up  = away
    elif home_margin < 0:
        champion   = away
        runner_up  = home
    else:
        # Shootout or draw — check shootout winner
        if pd.notna(final.get('shootout_winner')):
            champion  = final['shootout_winner']
            runner_up = away if champion == home else home
        else:
            # True draw with no shootout data — skip
            continue

    podium_records.append({
        'tournament': tournament,
        'year': year,
        'team': champion,
        'finish': 1
    })
    podium_records.append({
        'tournament': tournament,
        'year': year,
        'team': runner_up,
        'finish': 2
    })

    # Third place: look for a match on the day before the final
    # between two teams who are NOT in the final
    finalist_set = {home, away}
    pre_final = group[group['date'] < last_date]

    if len(pre_final) > 0:
        last_semifinal_date = pre_final['date'].max()
        third_place_candidates = pre_final[
            (pre_final['date'] == last_semifinal_date) &
            (~pre_final['home_team'].isin(finalist_set)) &
            (~pre_final['away_team'].isin(finalist_set))
        ]

        if len(third_place_candidates) > 0:
            tp_game = third_place_candidates.iloc[-1]
            tp_home = tp_game['home_team']
            tp_away = tp_game['away_team']
            tp_margin = tp_game['margin_home']

            if tp_margin > 0:
                third = tp_home
            elif tp_margin < 0:
                third = tp_away
            else:
                if pd.notna(tp_game.get('shootout_winner')):
                    third = tp_game['shootout_winner']
                else:
                    third = None

            if third:
                podium_records.append({
                    'tournament': tournament,
                    'year': year,
                    'team': third,
                    'finish': 3
                })

podium_flags_df = pd.DataFrame(podium_records)
podium_flags_df.to_csv('tournament_podiums.csv', index=False)
print(f"tournament_podiums.csv saved! ({len(podium_flags_df)} podium records)")
print(podium_flags_df.head(20).to_string())

# ============================================================
# STEP 10 - MERGE INTO FINAL OUTPUT FILE
# ============================================================

print("Building final output file...")

final_df = messi_df.copy()
final_df.rename(columns={'ranking_date': 'date'}, inplace=True)
final_df['year'] = final_df['season'].fillna(0).astype(int)

# Merge MESSI2 and MESSI3 ratings via merge_asof on (date, name).
# Each model has its OWN ranking_id sequence (different game-day sets), so
# merging by ranking_id is incorrect. merge_asof forward-fills the most
# recent prior MESSI2/MESSI3 rating onto each MESSI1 snapshot date.
final_df['date'] = pd.to_datetime(final_df['date'])

messi2_slim = messi2_df[['ranking_date', 'name', 'rating2', 'rank2', 'games_played2']].copy()
messi2_slim['ranking_date'] = pd.to_datetime(messi2_slim['ranking_date'])
messi2_slim = messi2_slim.sort_values('ranking_date')

messi3_slim = messi3_df[['ranking_date', 'name', 'rating3', 'rank3', 'games_played3']].copy()
messi3_slim['ranking_date'] = pd.to_datetime(messi3_slim['ranking_date'])
messi3_slim = messi3_slim.sort_values('ranking_date')

final_df = final_df.sort_values('date')

final_df = pd.merge_asof(
    final_df, messi2_slim,
    left_on='date', right_on='ranking_date',
    by='name', direction='backward',
)
final_df.drop(columns=['ranking_date'], inplace=True)

final_df = pd.merge_asof(
    final_df, messi3_slim,
    left_on='date', right_on='ranking_date',
    by='name', direction='backward',
)
final_df.drop(columns=['ranking_date'], inplace=True)

# Add confederation (must happen before name is renamed to country)
final_df['confederation'] = final_df['name'].map(CONFEDERATION_MAP).fillna('Unknown')

# Add most recent snapshot flag
latest_id = final_df['ranking_id'].max()
final_df['most_recent'] = np.where(final_df['ranking_id'] == latest_id, 1, 0)

# Add last match and last match date via merge_asof
# Carries forward each team's most recent result to every subsequent snapshot date
final_df['date'] = pd.to_datetime(final_df['date'])
lastmatch_df_sorted = lastmatch_df.copy()
lastmatch_df_sorted['date'] = pd.to_datetime(lastmatch_df_sorted['date'])
lastmatch_df_sorted = lastmatch_df_sorted.sort_values('date')

final_df = final_df.sort_values('date')
final_df = pd.merge_asof(
    final_df,
    lastmatch_df_sorted.rename(columns={'date': 'match_date'}),
    left_on='date',
    right_on='match_date',
    by=['name', 'season'],
    direction='backward'
)
final_df['last_match'] = final_df['last_match'].fillna('')
final_df['last_match_date'] = final_df['match_date'].dt.date
final_df.drop(columns=['match_date'], inplace=True)
final_df['date'] = final_df['date'].dt.date

# Flag rows where the rating date is a game day for that country
final_df['is_game_day'] = np.where(final_df['date'] == final_df['last_match_date'], 1, 0)

# Rename for clarity
final_df.rename(columns={'name': 'country'}, inplace=True)

# MESSI Blend — weighted average with scale adjustment
# MESSI1 is scaled by 0.65 to account for its higher absolute values
# (result of tournament up-weighting and smaller network) before blending
# Effective weight: 75% MESSI1 signal, 25% MESSI2 signal on comparable scales
# Z-score normalized blend: each model standardized within its snapshot first,
# then equal-weighted averaged. Standardization makes the three models
# scale-comparable so no single model's outliers (e.g. MESSI3 Cuba 1996) dominate.
for _col in ('rating', 'rating2', 'rating3'):
    grp = final_df.groupby('ranking_id')[_col]
    final_df[_col + '_z'] = (final_df[_col] - grp.transform('mean')) / grp.transform('std')

final_df['rating_blend'] = final_df[['rating_z', 'rating2_z', 'rating3_z']].mean(axis=1, skipna=True)
final_df['rank_blend'] = (
    final_df[final_df['rating_blend'].notna()]
    .groupby('ranking_id')['rating_blend']
    .rank(ascending=False, method='min')
)
final_df['rank_blend'] = final_df['rank_blend'].astype('Int64')

# ============================================================
# STEP 11 - ATTACH PODIUM FLAGS TO FINAL FILE
# ============================================================
# Simplified to a single tournament_finish column: 1st / 2nd / 3rd / blank
# Takes the best finish if a country has multiple in the same year

finish_map = {1: '1st', 2: '2nd', 3: '3rd'}
podium_flags_df['finish_label'] = podium_flags_df['finish'].map(finish_map)

# Keep only best finish per team per year (1st > 2nd > 3rd)
podium_best = (
    podium_flags_df
    .sort_values('finish')
    .drop_duplicates(subset=['team', 'year'], keep='first')
    [['team', 'year', 'finish_label']]
    .rename(columns={'team': 'country', 'year': 'year', 'finish_label': 'tournament_finish'})
)

final_df = pd.merge(final_df, podium_best, on=['country', 'year'], how='left')
final_df['tournament_finish'] = final_df['tournament_finish'].fillna('')

# ============================================================
# STEP 12 - WORLD CUP FINAL DAY FLAG
# ============================================================
# Flag every snapshot that falls on the last day of a FIFA World Cup edition.
# Lets you filter to post-World Cup rankings snapshots easily.

wc_df = df[df['tournament'] == 'FIFA World Cup'].copy()
wc_final_dates = (
    wc_df.groupby(wc_df['date'].dt.year)['date']
    .max()
    .reset_index(drop=True)
)
wc_final_date_set = set(wc_final_dates.dt.date)
final_df['is_world_cup_final_day'] = np.where(
    final_df['date'].isin(wc_final_date_set), 1, 0
)

# Final column order
final_df = final_df[[
    'ranking_id', 'date', 'year', 'country',
    'confederation',
    'rating',  'rank',                                           # MESSI1 (competitive)
    'rating2', 'rank2',                                          # MESSI2 (all matches)
    'rating3', 'rank3',                                          # MESSI3 (WC qualifying + finals only)
    'rating_blend', 'rank_blend',
    'games_played',
    'last_match_date', 'last_match', 'is_game_day',
    'most_recent', 'is_world_cup_final_day',
    'tournament_finish'
]]

final_df.sort_values(['ranking_id', 'rank'], inplace=True)
final_df.drop_duplicates(keep='first', inplace=True)

# Filter out pre-1986 data — insufficient network density for reliable ratings
final_df = final_df[final_df['date'] >= pd.to_datetime('1986-01-01').date()]

# min_games filter — drop teams with fewer than min_games competitive games in window.
# Mirrors ZIDANE: prevents stale ratings from inactive/banned teams from showing up
# (e.g., Mexico 1990 — banned, no games in 1990 but old games still in calendar window).
final_df = final_df[final_df['games_played'] >= min_games]
print(f"After min_games={min_games} filter: {len(final_df)} rows")

# ── Recentering removed ───────────────────────────────────────────────────────
# We previously shifted ratings by the WC-qualifier mean per snapshot to make
# the anchor "average WC qualifier = 0". But the rest of the network is so
# much weaker than that anchor that this pushed most of the dataset into
# strongly negative territory and the displayed values lost intuitive meaning.
# Reverting to raw Massey output. GOAT-level anomalies (small-confederation
# teams with sky-high ratings from a few blowout games) are now handled by
# a championship gate in generate_data.py instead.

final_df.to_csv('messi_ratings_final.csv', index=False)
print("messi_ratings_final.csv saved!")
print(f"\nTotal rows in final output: {len(final_df)}")
print(f"\nMost recent ratings snapshot (sorted by MESSI1 rank):")
print(final_df[final_df['most_recent'] == 1][
    ['rank', 'country', 'confederation', 'rating', 'rank2', 'rating2', 'rank_blend', 'rating_blend', 'games_played', 'last_match_date']
].head(20).to_string(index=False))

print(f"\nMost recent ratings snapshot (sorted by MESSI2 rank):")
print(final_df[final_df['most_recent'] == 1].sort_values('rank2')[
    ['rank2', 'country', 'confederation', 'rating2', 'rank', 'rating', 'rank_blend', 'rating_blend', 'games_played', 'last_match_date']
].head(20).to_string(index=False))

print(f"\nMost recent ratings snapshot (sorted by BLEND rank):")
print(final_df[final_df['most_recent'] == 1].sort_values('rank_blend')[
    ['rank_blend', 'country', 'confederation', 'rating_blend', 'rank', 'rating', 'rank2', 'rating2', 'games_played', 'last_match_date']
].head(20).to_string(index=False))

# World Cup final day — all three models
print(f"\nMESSI1 — World Cup final day snapshots (top 5 per edition):")
wc_snap = final_df[final_df['is_world_cup_final_day'] == 1].copy()
for dt in sorted(wc_snap['date'].unique()):
    print(f"\n  {dt}:")
    print(wc_snap[wc_snap['date'] == dt][['rank', 'country', 'rating']].head(5).to_string(index=False))

print(f"\nMESSI2 — World Cup final day snapshots (top 5 per edition):")
for dt in sorted(wc_snap['date'].unique()):
    print(f"\n  {dt}:")
    print(wc_snap[wc_snap['date'] == dt].sort_values('rank2')[['rank2', 'country', 'rating2']].head(5).to_string(index=False))

print(f"\nMESSI BLEND — World Cup final day snapshots (top 5 per edition):")
for dt in sorted(wc_snap['date'].unique()):
    print(f"\n  {dt}:")
    print(wc_snap[wc_snap['date'] == dt].sort_values('rank_blend')[['rank_blend', 'country', 'rating_blend']].head(5).to_string(index=False))


