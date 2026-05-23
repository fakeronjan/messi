# ============================================================
# MESSI - International Soccer Power Rankings
# Massey Elo Soccer Strength Index
# Based on ZIDANE / COBI architecture (homebrew WLS solver)
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS
# ============================================================

start_date         = '1980-01-01'   # first date to include in dataset
window_game_days   = 200            # rolling game-day window
margin_cap         = 4              # max goal margin (matches ZIDANE / COBI soccer family)
shootout_margin    = 0.5            # margin assigned to a shootout win
home_field_adv     = 0.5            # per-game HCA on raw goal margin
min_competitive_games = 5           # minimum NON-FRIENDLY games in window to appear in
                                    # final output. Friendlies still contribute to the
                                    # rating regression (cross-confederation signal) but
                                    # don't count toward eligibility. Fixes the Israel-1998
                                    # anomaly where 5 friendlies inflated a team to #1.
friendly_weight    = 0.25           # WLS observation weight applied to friendlies
                                    # (competitive games default to 1.0; tournament
                                    # tier uplift is multiplied on top via TOURNAMENT_WEIGHTS)

# WLS: weights affect observation influence, not margin magnitude.
# Margin transform (cap=4) + per-game HCA + shootout handling all applied
# UPSTREAM. The solver takes pre-prepped adj_margin_home as response, and the
# combined weight (recency × tournament tier × match type) as observation weight.
WEIGHTING_MODE = "wls"

# Re-process the most recent N ranking_ids (game-days) on every run so late-
# arriving data is absorbed. International games can post hours after final
# whistle and FIFA windows sometimes settle over multiple days.
RECOMPUTE_TAIL_DAYS = 7

data_url      = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
shootouts_url = 'https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv'

# ============================================================
# TOURNAMENT CONFIGURATION
# ============================================================

# WLS observation-weight uplifts for higher-tier tournaments. NOT applied
# to margin — these encode "this game is a more reliable signal of team
# strength" via observation weight in the WLS solve. Everything not listed
# defaults to 1.0x (qualifiers, regional cups, FIFA Series, etc.).
TOURNAMENT_WEIGHTS = {
    # Tier 1 — 2.0x: global pinnacle
    'FIFA World Cup':                    2.0,

    # Tier 2 — 1.5x: each confederation's peak championship (finals tournaments)
    'UEFA Euro':                         1.5,
    'Copa América':                      1.5,
    'African Cup of Nations':            1.5,
    'AFC Asian Cup':                     1.5,
    'Gold Cup':                          1.5,
    'CONCACAF Championship':             1.5,
    'Oceania Nations Cup':               1.5,

    # Tier 3 — 1.25x: marquee one-offs + biennial nations leagues
    'FIFA Confederations Cup':           1.25,
    'CONMEBOL–UEFA Cup of Champions':    1.25,
    'UEFA Nations League':               1.25,
    'CONCACAF Nations League':           1.25,
}

# Knockout tournaments with a final + third-place match — used for podium logic.
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

# Allowlist of legitimate competitive tournaments. Anything NOT in this list
# (and not in U23_TOURNAMENTS below) is treated as a friendly and gets
# friendly_weight applied to its WLS observation weight.
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
    'CAFA Nations Cup',

    # OFC
    'Oceania Nations Cup',
    'Oceania Nations Cup qualification',
    "MSG Prime Minister's Cup",
    'Melanesia Cup',
}

# U-23 events: not senior national-team lineups — exclude entirely.
U23_TOURNAMENTS = {
    'Asian Games', 'Southeast Asian Games', 'South Asian Games',
    'East Asian Games', 'Pacific Games', 'South Pacific Games',
    'South Pacific Mini Games', 'Pacific Mini Games',
    'All-African Games', 'Afro-Asian Games', 'Inter Games',
    'Olympic Games',
}

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


def _solve_massey(window_df):
    """
    Homebrew weighted-least-squares Massey solver. Mirrors ZIDANE / COBI.

    Takes a window df with home_team, away_team, adj_margin_home (HCA + cap
    pre-applied margin from home perspective), and weight (recency × tier ×
    match-type combined). Returns DataFrame with columns: name, rating, rank.

    Soccer-specific transforms (shootout handling, cap clip, per-game HFA)
    are applied UPSTREAM in the data-prep step. The solver just does WLS
    on the pre-prepped response variable.
    """
    teams = sorted(set(window_df["home_team"]) | set(window_df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    adj_margin = window_df["adj_margin_home"].to_numpy(dtype=float)
    weights    = window_df["weight"].to_numpy(dtype=float)
    home_names = window_df["home_team"].to_numpy()
    away_names = window_df["away_team"].to_numpy()

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] =  1.0
        X[i, team_idx[away_names[i]]] = -1.0

    y[:n_games] = adj_margin
    w[:n_games] = weights

    # Zero-sum constraint via high-weight extra row.
    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


# ============================================================
# STEP 1 - LOAD AND CLEAN THE RAW DATA
# ============================================================

print("Loading results from GitHub...")
raw_df = pd.read_csv(data_url)
shootouts_df = pd.read_csv(shootouts_url)

print(f"Raw results loaded: {len(raw_df)} rows")
print(f"Shootouts loaded: {len(shootouts_df)} rows")

raw_df['date'] = pd.to_datetime(raw_df['date'])
shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])

# Date filter + drop U-23 events. Everything else (competitive + friendlies)
# stays in the dataset — friendlies will get downweighted via WLS observation
# weight rather than excluded outright.
df = raw_df[
    (raw_df['date'] >= pd.to_datetime(start_date)) &
    (~raw_df['tournament'].isin(U23_TOURNAMENTS))
].copy()

print(f"After date filter and U-23 exclusion: {len(df)} rows")

# ============================================================
# STEP 2 - HANDLE SHOOTOUTS
# ============================================================
# results.csv scores reflect full time + extra time. A 0-0 AET game
# decided on penalties needs the shootout winner credited a ±0.5 margin.

shootouts_df = shootouts_df[['date', 'home_team', 'away_team', 'winner']].copy()
shootouts_df.columns = ['date', 'home_team', 'away_team', 'shootout_winner']
df = pd.merge(df, shootouts_df, on=['date', 'home_team', 'away_team'], how='left')

# ============================================================
# STEP 3 - CALCULATE MARGINS
# ============================================================

df['raw_margin_home'] = df['home_score'] - df['away_score']
df['raw_margin_away'] = -df['raw_margin_home']

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

df['margin_home'] = df['margin_home'].clip(-margin_cap, margin_cap)
df['margin_away'] = -df['margin_home']

# ============================================================
# STEP 4 - HOME FIELD ADJUSTMENT
# ============================================================
# Per-game HFA on raw goal margin (pre-transform). Neutral venues get no
# adjustment — WC finals, CFP-style neutrals, international kickoffs.

df['hfa'] = np.where(df['neutral'] == True, 0, home_field_adv)
df['adj_margin_home'] = df['margin_home'] - df['hfa']
df['adj_margin_away'] = -df['adj_margin_home']

# ============================================================
# STEP 4b - MATCH TYPE + TOURNAMENT TIER TAGS
# ============================================================
# match_type: 'competitive' if in COMPETITIVE_TOURNAMENTS allowlist,
#             else 'friendly'.
# tier_weight: 1.5/1.25/1.0 per TOURNAMENT_WEIGHTS (default 1.0).
# match_type_weight: 1.0 for competitive, friendly_weight for friendlies.
#
# The two combine multiplicatively at solve time with the recency weight
# to form the WLS observation weight for each game.

df['match_type'] = np.where(
    df['tournament'].isin(COMPETITIVE_TOURNAMENTS),
    'competitive',
    'friendly'
)
df['match_type_weight'] = np.where(df['match_type'] == 'competitive', 1.0, friendly_weight)
df['tier_weight'] = df['tournament'].map(TOURNAMENT_WEIGHTS).fillna(1.0)

# ============================================================
# STEP 5 - WINS FOR STANDINGS
# ============================================================

df['home_win'] = np.where(
    df['margin_home'] > 0, 1,
    np.where(df['margin_home'] == 0, 0.5, 0)
)
df['away_win'] = 1 - df['home_win']

# Shootout overrides — winner gets full W, loser full L (for standings only,
# margin is already set to ±shootout_margin for the rating solve)
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'home_win'] = 1
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'away_win'] = 0
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'home_win'] = 0
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'away_win'] = 1

# ============================================================
# STEP 6 - CONFEDERATION TAGS
# ============================================================

df['home_confederation'] = df['home_team'].map(CONFEDERATION_MAP).fillna('Unknown')
df['away_confederation'] = df['away_team'].map(CONFEDERATION_MAP).fillna('Unknown')

unknown_teams = set(
    df.loc[df['home_confederation'] == 'Unknown', 'home_team'].tolist() +
    df.loc[df['away_confederation'] == 'Unknown', 'away_team'].tolist()
)
if unknown_teams:
    print(f"WARNING: {len(unknown_teams)} non-FIFA teams found (will be dropped):")
    for t in sorted(unknown_teams)[:20]:
        print(f"  - {t}")
    if len(unknown_teams) > 20:
        print(f"  ... and {len(unknown_teams) - 20} more")

# Drop matches involving non-FIFA "nations" (CONIFA / Viva World Cup / ELF
# Cup teams: Ellan Vannin, Padania, Kárpátalja, Northern Cyprus, etc.).
# They form a disconnected subgraph from the FIFA network, which causes the
# WLS solver to produce arbitrary high ratings for them under the zero-sum
# constraint. CONFEDERATION_MAP is the FIFA-member allowlist.
_n_before = len(df)
df = df[
    (df['home_confederation'] != 'Unknown') &
    (df['away_confederation'] != 'Unknown')
].copy()
print(f"After non-FIFA team drop: {len(df)} rows ({_n_before - len(df)} matches removed)")

# ============================================================
# STEP 7 - DATE IDs FOR ROLLING WINDOW
# ============================================================

df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

df['grouped_date_id'] = df.groupby('date').ngroup() + 1
df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

df.to_csv('all_soccer_games.csv', index=False)
print("Master game CSV saved: all_soccer_games.csv")

# ============================================================
# STEP 7b - LAST MATCH STRINGS
# ============================================================
# Format: "W vs. France 2-1 (UEFA Euro)" or "L @ Brazil 0-3 (Friendly)"
# Includes friendlies — fans want to see the most recent game played
# regardless of competitiveness.

df['home_score_int'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score_int'] = pd.to_numeric(df['away_score'], errors='coerce')
df = df.dropna(subset=['home_score_int', 'away_score_int']).copy()
df['home_score_int'] = df['home_score_int'].astype(int)
df['away_score_int'] = df['away_score_int'].astype(int)

df['home_result_flag'] = np.where(
    df['home_win'] == 1, 'W',
    np.where(df['home_win'] == 0.5, 'D', 'L')
)
df['away_result_flag'] = np.where(
    df['away_win'] == 1, 'W',
    np.where(df['away_win'] == 0.5, 'D', 'L')
)

df['home_venue'] = np.where(df['neutral'] == True, ' vs. (N) ', ' vs. ')
df['away_venue'] = np.where(df['neutral'] == True, ' vs. (N) ', ' @ ')

df['home_last_match'] = (
    df['home_result_flag'] +
    df['home_venue'] +
    df['away_team'] + ' ' +
    df['home_score_int'].map(str) + '-' + df['away_score_int'].map(str) +
    ' (' + df['tournament'] + ')'
)
df['away_last_match'] = (
    df['away_result_flag'] +
    df['away_venue'] +
    df['home_team'] + ' ' +
    df['away_score_int'].map(str) + '-' + df['home_score_int'].map(str) +
    ' (' + df['tournament'] + ')'
)

lastmatch_home = df[['date', 'home_team', 'home_last_match']].copy()
lastmatch_home.columns = ['date', 'name', 'last_match']
lastmatch_away = df[['date', 'away_team', 'away_last_match']].copy()
lastmatch_away.columns = ['date', 'name', 'last_match']
lastmatch_df = pd.concat([lastmatch_home, lastmatch_away], axis=0).reset_index(drop=True)
lastmatch_df['date'] = pd.to_datetime(lastmatch_df['date']).dt.date
lastmatch_df['season'] = pd.to_datetime(lastmatch_df['date']).dt.year.astype('int64')

# ============================================================
# STEP 8 - ROLLING WLS RATINGS (MESSI)
# ============================================================
# One snapshot per game-day, rolling window_game_days game-days. WLS
# observation weight per game = recency × tournament tier × match type.
# Incremental: skips date IDs already in messi_ratings.csv.gz cache.
# Cache is gzipped per the soccer-family convention (would exceed
# GitHub's 50MB recommendation uncompressed).

print("Starting MESSI rating calculations...")

max_date_id = int(df['grouped_date_id'].max())
min_date_id = 1

try:
    messi_df = pd.read_csv('messi_ratings.csv.gz')
    all_ids = sorted(messi_df['ranking_id'].unique())
    if len(all_ids) > RECOMPUTE_TAIL_DAYS:
        tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
        n_dropped = int((messi_df['ranking_id'] >= tail_threshold).sum())
        messi_df = messi_df[messi_df['ranking_id'] < tail_threshold].copy()
        print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
              f"({n_dropped:,} rows dropped from cache for late-arriving-data refresh)")
    max_id_ranked = int(messi_df['ranking_id'].max()) if not messi_df.empty else -1
    min_id_ranked = int(messi_df['ranking_id'].min()) if not messi_df.empty else -1
    if max_id_ranked >= 0:
        print(f"Existing ratings found. Ranked IDs: {min_id_ranked} to {max_id_ranked}")
except FileNotFoundError:
    messi_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_id_ranked = -1
    min_id_ranked = -1
    print("No existing ratings found - running full history from scratch.")

last_printed_ym = None
for i in range(min_date_id, max_date_id + 1):

    if min_id_ranked <= i <= max_id_ranked:
        continue

    current_date = df.loc[df['grouped_date_id'] == i, 'date'].max()
    if pd.isnull(current_date):
        continue

    working_df = df.loc[
        (df['grouped_date_id'] >= i - window_game_days + 1) &
        (df['grouped_date_id'] <= i)
    ].copy()

    if len(working_df) < 10:
        continue

    # Combined WLS observation weight: recency × tier × match_type
    working_df['game_days_ago'] = i - working_df['grouped_date_id']
    working_df['date_weight'] = 1 - (working_df['game_days_ago'] / window_game_days)
    working_df['weight'] = (
        working_df['date_weight'] *
        working_df['tier_weight'] *
        working_df['match_type_weight']
    )

    # Drop zero-weight rows and 0-margin regulation draws (shootout-decided
    # 0-0 games have ±shootout_margin and stay in). Matches the prior rankit
    # filter; the solver itself doesn't need this, but it keeps the network
    # density definition consistent with the pre-port pipeline.
    working_df = working_df[(working_df['weight'] > 0) & (working_df['adj_margin_home'] != 0)]
    if len(working_df) < 10:
        continue

    season = current_date.year
    current_ym = current_date.strftime('%Y-%m')
    if current_ym != last_printed_ym:
        pct = round(100 * i / max_date_id)
        print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct}% complete)")
        last_printed_ym = current_ym

    try:
        ranked = _solve_massey(working_df)
        if ranked['rating'].isna().any() or np.isinf(ranked['rating']).any():
            continue

        ranked['ranking_id']   = i
        ranked['ranking_date'] = current_date.date()
        ranked['season']       = season

        home_gp = working_df.groupby('home_team').size().reset_index(name='gp_home')
        away_gp = working_df.groupby('away_team').size().reset_index(name='gp_away')
        home_gp.columns = ['name', 'gp_home']
        away_gp.columns = ['name', 'gp_away']
        gp = pd.merge(home_gp, away_gp, on='name', how='outer').fillna(0)
        gp['games_played'] = (gp['gp_home'] + gp['gp_away']).astype(int)

        # Competitive (non-friendly) game counts feed the eligibility filter.
        comp_df = working_df[working_df['match_type'] == 'competitive']
        comp_home = comp_df.groupby('home_team').size().reset_index(name='cgp_home')
        comp_away = comp_df.groupby('away_team').size().reset_index(name='cgp_away')
        comp_home.columns = ['name', 'cgp_home']
        comp_away.columns = ['name', 'cgp_away']
        cgp = pd.merge(comp_home, comp_away, on='name', how='outer').fillna(0)
        cgp['competitive_games_played'] = (cgp['cgp_home'] + cgp['cgp_away']).astype(int)
        gp = pd.merge(gp, cgp[['name', 'competitive_games_played']], on='name', how='left')
        gp['competitive_games_played'] = gp['competitive_games_played'].fillna(0).astype(int)

        ranked = pd.merge(ranked, gp[['name', 'games_played', 'competitive_games_played']], on='name', how='left')
        ranked['games_played'] = ranked['games_played'].fillna(0).astype(int)
        ranked['competitive_games_played'] = ranked['competitive_games_played'].fillna(0).astype(int)

        messi_df = pd.concat([messi_df, ranked], axis=0, sort=False).reset_index(drop=True)

    except Exception as e:
        print(f"  [skip] date_id {i}: {e}")
        continue

messi_df.sort_values(['ranking_id', 'name'], inplace=True)
messi_df.drop_duplicates(keep='first', inplace=True)
messi_df['ranking_date'] = pd.to_datetime(messi_df['ranking_date']).dt.date
messi_df.to_csv('messi_ratings.csv.gz', index=False, compression='gzip')
print(f"messi_ratings.csv.gz saved ({len(messi_df):,} rows)")

# ============================================================
# STEP 9 - TOURNAMENT PODIUM FLAGS
# ============================================================
# For each edition of every PODIUM_TOURNAMENT, identify 1st, 2nd, and 3rd.
# Final = match(es) on the last date of the tournament. Bracket-walk
# disambiguates same-day final vs 3rd-place playoff (Africa, Oceania, etc.).

print("Calculating tournament podium flags...")

podium_df = df[df['tournament'].isin(PODIUM_TOURNAMENTS)].copy()
podium_df['year'] = podium_df['date'].dt.year

podium_records = []

for (tournament, year), group in podium_df.groupby(['tournament', 'year']):

    group = group.sort_values('date')
    last_date = group['date'].max()

    final_games = group[group['date'] == last_date]

    if len(final_games) == 0:
        continue

    # Same-day final vs 3rd-place playoff disambiguation: in the final,
    # both teams won their previous match (semis); in the 3rd-place
    # playoff, both teams lost their previous match.
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
        # Bracket-walk failed (in-progress tournament, data hasn't reached
        # knockouts yet). Only fall back to "last game on last date" if the
        # tournament is conclusively complete (≥14 days past latest match).
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
        if pd.notna(final.get('shootout_winner')):
            champion  = final['shootout_winner']
            runner_up = away if champion == home else home
        else:
            continue

    podium_records.append({'tournament': tournament, 'year': year, 'team': champion,  'finish': 1})
    podium_records.append({'tournament': tournament, 'year': year, 'team': runner_up, 'finish': 2})

    # Third place: match on day before final between non-finalists
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
                podium_records.append({'tournament': tournament, 'year': year, 'team': third, 'finish': 3})

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
final_df['season'] = final_df['season'].astype('int64')   # ensure merge_asof key dtype matches lastmatch_df
final_df['date'] = pd.to_datetime(final_df['date'])

# Confederation
final_df['confederation'] = final_df['name'].map(CONFEDERATION_MAP).fillna('Unknown')

# Most-recent snapshot flag
latest_id = final_df['ranking_id'].max()
final_df['most_recent'] = np.where(final_df['ranking_id'] == latest_id, 1, 0)

# Last match + last match date via merge_asof, scoped by (name, season)
# so the start of a new calendar year correctly shows empty for teams
# that haven't played yet.
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

final_df.rename(columns={'name': 'country'}, inplace=True)

# ============================================================
# STEP 11 - ATTACH PODIUM FLAGS TO FINAL FILE
# ============================================================
# Single tournament_finish column: '1st' / '2nd' / '3rd' / blank.
# Best finish wins if a country has multiple in the same year.

finish_map = {1: '1st', 2: '2nd', 3: '3rd'}
podium_flags_df['finish_label'] = podium_flags_df['finish'].map(finish_map)

podium_best = (
    podium_flags_df
    .sort_values('finish')
    .drop_duplicates(subset=['team', 'year'], keep='first')
    [['team', 'year', 'finish_label']]
    .rename(columns={'team': 'country', 'finish_label': 'tournament_finish'})
)

final_df = pd.merge(final_df, podium_best, on=['country', 'year'], how='left')
final_df['tournament_finish'] = final_df['tournament_finish'].fillna('')

# ============================================================
# STEP 12 - WORLD CUP FINAL DAY FLAG
# ============================================================

wc_df = df[df['tournament'] == 'FIFA World Cup'].copy()
wc_final_dates = (
    wc_df.groupby(wc_df['date'].dt.year)['date'].max().reset_index(drop=True)
)
wc_final_date_set = set(wc_final_dates.dt.date)
final_df['is_world_cup_final_day'] = np.where(
    final_df['date'].isin(wc_final_date_set), 1, 0
)

# Final column order
final_df = final_df[[
    'ranking_id', 'date', 'year', 'country',
    'confederation',
    'rating', 'rank',
    'games_played', 'competitive_games_played',
    'last_match_date', 'last_match', 'is_game_day',
    'most_recent', 'is_world_cup_final_day',
    'tournament_finish'
]]

final_df.sort_values(['ranking_id', 'rank'], inplace=True)
final_df.drop_duplicates(keep='first', inplace=True)

# 1986+ cutoff (data quality cliff before — Maradona-era WC anchor)
final_df = final_df[final_df['date'] >= pd.to_datetime('1986-01-01').date()]

# Eligibility filter: require min_competitive_games NON-FRIENDLY games in window.
# Friendlies still contribute to the regression (cross-confederation signal)
# but don't count toward eligibility. This fixes the Israel-1998 hot-streak
# anomaly where 5 friendlies inflated a team to #1.
final_df = final_df[final_df['competitive_games_played'] >= min_competitive_games]
print(f"After min_competitive_games={min_competitive_games} filter: {len(final_df)} rows")

final_df.to_csv('messi_ratings_final.csv', index=False)
print("messi_ratings_final.csv saved!")
print(f"\nTotal rows in final output: {len(final_df)}")
print(f"\nMost recent ratings snapshot (top 20):")
print(final_df[final_df['most_recent'] == 1].sort_values('rank')[
    ['rank', 'country', 'confederation', 'rating', 'games_played', 'last_match_date']
].head(20).to_string(index=False))

print(f"\nWorld Cup final day snapshots (top 5 per edition):")
wc_snap = final_df[final_df['is_world_cup_final_day'] == 1].copy()
for dt in sorted(wc_snap['date'].unique()):
    print(f"\n  {dt}:")
    print(wc_snap[wc_snap['date'] == dt].sort_values('rank')[
        ['rank', 'country', 'rating']
    ].head(5).to_string(index=False))
