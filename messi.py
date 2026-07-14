# ============================================================
# MESSI - International Soccer Power Rankings
# fakeronjan WLS Elo Soccer Strength Index
# Based on ZIDANE / COBI architecture (homebrew WLS solver)
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS
# ============================================================

start_date         = '1980-01-01'   # first date to include in dataset
window_years       = 3              # rolling CALENDAR-year window for team ratings (changed
                                    # 2026-06-07 from a 200 game-day window). Calendar years
                                    # keep the horizon consistent regardless of game-day
                                    # density / scheduling, which fits the lumpy international
                                    # calendar. The old 200-day window collapsed idle elites
                                    # during pre-tournament friendly lulls (Argentina #7->#19
                                    # without a loss) because their competitive pedigree aged
                                    # out. 3yr retains pedigree; tested 5yr but it re-propped
                                    # Brazil (padded blowouts accumulate). Recency half-life
                                    # (below) does the responsiveness work, decoupled from the
                                    # window length.
recency_halflife_years = 1.5        # exponential recency half-life within the window. Recent
                                    # form dominates; a long light-weighted tail of pedigree
                                    # keeps proven teams from evaporating during lulls.
margin_cap         = 4              # ZIDANE/COBI soccer-family default. Was briefly 8 (2026-05-30, "margin signal headroom") but that raise PREDATED the confederation calibration (2026-06-07) and, with nothing to counterbalance it, let minnow blowouts inflate isolated blocs (OFC/Australia #1 in 2004). Reverted to 4 on 2026-07-05: the calibration now carries part of the SOS load, and past-4 goal margins are almost all minnow mismatches (only 2.76% of games exceed |5|)
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

# ── Offense/defense split (added 2026-06-26) ────────────────────────────────
# A companion solve decomposes each team's rating into an attacking rating
# (goals scored) and a defending rating (goals conceded), re-anchored so
# rating_o + rating_d equals the calibrated rating. See _solve_wls_od.
od_off_share       = 0.5            # share of each match's home edge attributed to attack
                                    # (the rest goes to defense). 0.5 = even split.
od_goal_cap        = 6              # per-side goal clip for the O/D lean only (the re-anchor
                                    # still pins O+D to the calibrated rating, so this touches
                                    # the attack-vs-defense lean, not the level). Without it,
                                    # historic routs (Australia 31-0 American Samoa, 2001 OFC
                                    # qualifying) hand minnows the top attacking leans. 6 tames
                                    # those while keeping real attacking dominance intact.

# ── Confederation calibration (added 2026-06-07) ────────────────────────────
# The match graph is 6 dense confederation clusters joined by thin bridges
# (mostly down-weighted friendlies). The WLS solve pins WITHIN-confederation
# ordering well, but the BETWEEN-confederation LEVEL rode on whatever cross-
# confed friendlies were recently in the window - so the whole UEFA bloc could
# lurch >1.0 in a day on two friendlies (Norway #5->#2 on 2026-03-26 without
# playing). Fix = a confed<->team iterative calibration: each team is shrunk
# toward its confederation's CALIBRATED level (partial pooling), and that level
# is a slow multi-year estimate, re-estimated from the solved ratings but
# anchored back to the slow value each pass. Within-confed reactivity is kept;
# the bloc level stops twitching. See docs/calibration_plan.md.
confed_prior_lambda      = 1.0       # partial-pooling strength: WLS weight on the
                                     # per-team prior pulling each team toward its
                                     # confederation level. Lowered 2.0->1.0 on 2026-06-07
                                     # with the move to a 3yr window: the longer window now
                                     # does most of the thin-team work (a team is rarely
                                     # thin-in-window), so a gentler prior suffices and
                                     # avoids over-shrinking idle elites toward the confed
                                     # mean. Thin teams still get pulled; rich teams barely.
confed_offset_years      = 8         # horizon for the slow cross-confederation level
confed_offset_halflife_d = 3 * 365   # recency half-life (days) within that horizon;
                                     # World Cup games (tier 2.0x) dominate the level
recenter_anchor_alpha    = 0.6       # in the confed re-estimation blend, weight on the
                                     # slow offset vs the solved bloc level (higher =
                                     # more lurch-proof, lower = more reactive level)
calibration_iters        = 3         # fixed K: confed -> team -> confed passes per
                                     # snapshot. Deterministic from games (caches cleanly)

# WLS: weights affect observation influence, not margin magnitude.
# Margin transform (cap=4) + per-game HCA + shootout handling all applied
# UPSTREAM. The solver takes pre-prepped adj_margin_home as response, and the
# combined weight (recency × tournament tier × match type) as observation weight.

# Re-process the most recent N ranking_ids (game-days) on every run so late-
# arriving data is absorbed. International games can post hours after final
# whistle and FIFA windows sometimes settle over multiple days.
RECOMPUTE_TAIL_DAYS = 7

data_url      = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
shootouts_url = 'https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv'

# ── Live-score overlay (football-data.org) ──────────────────────────────────
# martj42 is the system of record (full breadth + all history) but lags match
# days by 1-2 days while volunteers backfill scores. For an in-progress major
# that football-data.org covers, we overlay its FINISHED scores onto martj42's
# already-loaded fixture rows that still have empty scores. This ONLY fills
# blanks - martj42 values are never overwritten - and is gated to recent unplayed
# rows, so once a tournament finishes (and martj42 catches up) MESSI stops
# calling the API entirely. That gate is also why a cancelled/free key is fine
# off-tournament: no recent blanks -> no call. WC is football-data's free tier.
FD_BASE = 'https://api.football-data.org/v4'
# football-data competition code -> martj42 tournament name. Only the FINAL
# tournament is covered (NOT qualifiers), and only these competitions exist on
# the source. Add 'EC' -> 'UEFA Euro' in Euro years.
FD_LIVE_COMPETITIONS = {
    'WC': 'FIFA World Cup',
}
# football-data team name -> martj42 canonical name. Verified by diffing the two
# sources' WC-2026 rosters (48 teams each, 4 divergences). Extend if a future
# edition surfaces a new mismatch (the overlay logs any match it can't place).
FD_TEAM_ALIASES = {
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Cape Verde Islands':  'Cape Verde',
    'Congo DR':            'DR Congo',
    'Czechia':             'Czech Republic',
}
# How far back a blank fixture can be and still trigger an API call. Keeps the
# overlay scoped to an in-progress tournament; off-season there are no recent
# blanks so no request is made.
FD_OVERLAY_LOOKBACK_DAYS = 10
FD_OVERLAY_LOOKAHEAD_DAYS = 3

# Venue-city -> IANA timezone for the FIFA World Cup 2026 host cities, used to
# convert football-data.org's UTC kickoff into the LOCAL match day (the date
# martj42 and the site display by). martj42 occasionally ships a stale fixture
# date before a volunteer fixes it; FD's kickoff + this map is the authority.
# A late US-night kickoff (e.g. 00:00 UTC) can be the PREVIOUS local day, so the
# raw UTC date is never trusted directly. Country fallback for anything unmapped.
WC_VENUE_TZ = {
    'Arlington':       'America/Chicago',      'Atlanta':        'America/New_York',
    'Dallas':          'America/Chicago',      'East Rutherford': 'America/New_York',
    'Foxborough':      'America/New_York',     'Guadalupe':      'America/Monterrey',
    'Houston':         'America/Chicago',      'Inglewood':      'America/Los_Angeles',
    'Kansas City':     'America/Chicago',      'Mexico City':    'America/Mexico_City',
    'Miami Gardens':   'America/New_York',     'Philadelphia':   'America/New_York',
    'Santa Clara':     'America/Los_Angeles',  'Seattle':        'America/Los_Angeles',
    'Toronto':         'America/Toronto',      'Vancouver':      'America/Vancouver',
    'Zapopan':         'America/Mexico_City',
}
_COUNTRY_TZ_FALLBACK = {
    'United States': 'America/New_York', 'Canada': 'America/Toronto',
    'Mexico': 'America/Mexico_City',
}

# Append-only DB horizon (DUNCAN model): games older than this are LOCKED to the
# committed local DB - the daily martj42 re-fetch can no longer overwrite them,
# it can only add genuinely new history. Games within the horizon are the
# "active zone" (an in-progress tournament, recent friendlies/quals martj42 is
# still backfilling) where the fresh fetch + FD overlay stay authoritative so
# pending games finalize and live corrections land. Comfortably covers a full
# major (~40 days) plus martj42's 1-2 day lag.
DB_LOCK_DAYS = 60

# ============================================================
# TOURNAMENT CONFIGURATION
# ============================================================

# WLS observation-weight uplifts for higher-tier tournaments. NOT applied
# to margin - these encode "this game is a more reliable signal of team
# strength" via observation weight in the WLS solve. Everything not listed
# defaults to 1.0x (qualifiers, regional cups, FIFA Series, etc.).
TOURNAMENT_WEIGHTS = {
    # Tier 1 - 2.0x: global pinnacle
    'FIFA World Cup':                    2.0,

    # Tier 2 - 1.5x: each confederation's peak championship (finals tournaments)
    'UEFA Euro':                         1.5,
    'Copa América':                      1.5,
    'African Cup of Nations':            1.5,
    'AFC Asian Cup':                     1.5,
    'Gold Cup':                          1.5,
    'CONCACAF Championship':             1.5,
    'Oceania Nations Cup':               1.5,

    # Tier 3 - 1.25x: marquee one-offs + biennial nations leagues
    'Confederations Cup':                1.25,   # martj42 source string (NOT "FIFA Confederations Cup")
    'CONMEBOL–UEFA Cup of Champions':    1.25,
    'UEFA Nations League':               1.25,
    'CONCACAF Nations League':           1.25,
}

# Knockout tournaments with a final + third-place match - used for podium logic.
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
    'Confederations Cup',         # martj42 source string; discontinued 2017 but kept for historical podiums
]

# Locked-in final dates for in-progress editions whose knockout fixtures are NOT
# yet in the data feed (teams are TBD until the bracket fills, so the feed stops
# at the group stage). An edition with an entry here is NOT crowned until its
# final date has passed - the schedule alone can't prove it's over, since "no
# future games loaded" is indistinguishable from "knockouts not yet scheduled".
# This is what stops the podium bracket-walk from declaring a "champion" between
# the group stage ending and the knockout dates populating. Add the current
# major(s) each cycle; entries for concluded editions are harmless but prune for
# tidiness. (2026 WC final: 2026-07-19, MetLife Stadium.)
TOURNAMENT_END_DATES = {
    ('FIFA World Cup', 2026): date(2026, 7, 19),
}

# Allowlist of legitimate competitive tournaments. Anything NOT in this list
# (and not in U23_TOURNAMENTS below) is treated as a friendly and gets
# friendly_weight applied to its WLS observation weight.
COMPETITIVE_TOURNAMENTS = {
    # FIFA global
    'FIFA World Cup',
    'FIFA World Cup qualification',
    'Confederations Cup',   # martj42 source string (NOT "FIFA Confederations Cup") - senior FIFA event, must not fall through to friendly weight
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

# U-23 events: not senior national-team lineups - exclude entirely.
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

# National teams that changed FIFA confederation. Each entry lists the
# confederation in effect during [start, end] (inclusive, ISO dates), BEFORE the
# current one in CONFEDERATION_MAP; outside these ranges the current mapping
# applies. This flows into BOTH the cross-confederation ratings calibration and
# the displayed badges, so a team is rated/shown under the confederation it
# actually competed in at the time. (Same shape as generate_data's NAME_HISTORY.)
CONFEDERATION_HISTORY = {
    'Australia':  [('OFC', '1900-01-01', '2005-12-31')],   # OFC -> AFC, Jan 2006
    'Kazakhstan': [('AFC', '1991-12-16', '2001-12-31')],   # AFC -> UEFA, 2002
    'Israel':     [('OFC', '1900-01-01', '1993-12-31')],   # competed via OFC until UEFA membership, 1994
}


def _apply_confederation_history(frame, team_col, conf_col, date_col):
    """Override confederation with the as-of-date value for the few teams that
    switched confederation (vectorized; only touches the listed movers)."""
    if not CONFEDERATION_HISTORY:
        return
    dates = pd.to_datetime(frame[date_col])
    for team, periods in CONFEDERATION_HISTORY.items():
        is_team = frame[team_col] == team
        if not is_team.any():
            continue
        for conf, start, end in periods:
            m = is_team & (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            frame.loc[m, conf_col] = conf


def _solve_wls(window_df, confed_prior=None, prior_lambda=0.0):
    """
    Homebrew weighted-least-squares fakeronjan WLS solver. Mirrors ZIDANE / COBI.

    Takes a window df with home_team, away_team, adj_margin_home (HCA + cap
    pre-applied margin from home perspective), and weight (recency × tier ×
    match-type combined). Returns DataFrame with columns: name, rating, rank.

    Soccer-specific transforms (shootout handling, cap clip, per-game HFA)
    are applied UPSTREAM in the data-prep step. The solver just does WLS
    on the pre-prepped response variable.

    PARTIAL POOLING (Layer 1): when confed_prior {confederation: level} and
    prior_lambda > 0 are supplied, one extra observation row per team is added
    pulling that team's rating toward its confederation's level with weight
    prior_lambda. This is a ridge that shrinks toward the confed level (not 0):
    thin-evidence teams get dragged to "average team in their confederation"
    while well-observed teams override the prior and keep their earned rating.
    """
    teams = sorted(set(window_df["home_team"]) | set(window_df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    use_prior = bool(confed_prior) and prior_lambda > 0
    n_prior = n_teams if use_prior else 0
    n_rows = n_games + 1 + n_prior

    X = np.zeros((n_rows, n_teams))
    y = np.zeros(n_rows)
    w = np.zeros(n_rows)

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
    X[n_games, :] = 1.0
    y[n_games] = 0.0
    w[n_games] = 1.0e8

    # Per-team prior rows: shrink toward confederation level.
    if use_prior:
        for j, t in enumerate(teams):
            X[n_games + 1 + j, j] = 1.0
            y[n_games + 1 + j] = confed_prior.get(CONFEDERATION_MAP.get(t), 0.0)
            w[n_games + 1 + j] = prior_lambda

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _solve_wls_od(window_df, off_share=0.5, goal_cap=None):
    """
    Offense/defense companion to _solve_wls on the same window. Splits team
    strength into an attacking rating (goals scored vs an average opponent) and
    a defending rating (goals conceded vs an average opponent).

    Each match contributes two goal half-equations, from the home perspective:
        home_O - away_D = home_goals - mu - hfa * off_share
        away_O - home_D = away_goals - mu + hfa * (1 - off_share)
    where mu is the window mean goals per team per match, so attack and defense
    are each centered on zero, and the per-match hfa (0 at neutral venues) is
    split between the two sides by off_share.

    Parameter layout: [O_1..O_n, D_1..D_n], pinned by two zero-sum rows. A
    higher rating_o means more goals scored than an average team would; a higher
    rating_d means fewer conceded. A team's net contribution to goal margin is
    rating_o + rating_d, so the home-minus-away margin reproduces the single
    solver's (home - away) + hfa.

    The raw O/D level is NOT calibrated here. The caller re-anchors rating_o +
    rating_d to the confederation-calibrated rating, so the margin cap, shootout
    handling, and partial pooling carry through to the split; only the
    attack-vs-defense lean (rating_o - rating_d) comes from this solve.

    goal_cap, when set, clips each side's goals before the fit so one rout (e.g.
    a 9-0 qualifier) can't dominate a team's attacking lean.
    """
    teams = sorted(set(window_df["home_team"]) | set(window_df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    home_goals = window_df["home_score_int"].to_numpy(dtype=float)
    away_goals = window_df["away_score_int"].to_numpy(dtype=float)
    if goal_cap is not None:
        home_goals = np.clip(home_goals, 0, goal_cap)
        away_goals = np.clip(away_goals, 0, goal_cap)
    hfa        = window_df["hfa"].to_numpy(dtype=float)
    weights    = window_df["weight"].to_numpy(dtype=float)
    home_names = window_df["home_team"].to_numpy()
    away_names = window_df["away_team"].to_numpy()

    mu = (home_goals.sum() + away_goals.sum()) / (2 * n_games)
    off = float(off_share)
    deff = 1.0 - off

    # 2 rows per match + 2 zero-sum constraint rows.
    X = np.zeros((2 * n_games + 2, 2 * n_teams))
    y = np.zeros(2 * n_games + 2)
    w = np.zeros(2 * n_games + 2)

    for i in range(n_games):
        h = team_idx[home_names[i]]
        a = team_idx[away_names[i]]
        # home_O - away_D = home_goals - mu - hfa*off
        X[2*i,     h]           =  1.0
        X[2*i,     n_teams + a] = -1.0
        y[2*i]                  = home_goals[i] - mu - hfa[i] * off
        w[2*i]                  = weights[i]
        # away_O - home_D = away_goals - mu + hfa*deff
        X[2*i + 1, a]           =  1.0
        X[2*i + 1, n_teams + h] = -1.0
        y[2*i + 1]              = away_goals[i] - mu + hfa[i] * deff
        w[2*i + 1]              = weights[i]

    # Zero-sum on offense, then on defense.
    X[-2, :n_teams] = 1.0
    w[-2] = 1.0e8
    X[-1, n_teams:] = 1.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)

    return pd.DataFrame({
        "name":     teams,
        "rating_o": sol[:n_teams],
        "rating_d": sol[n_teams:],
    })


def _confed_offset(cross_win, as_of_date, halflife_days):
    """
    Layer 2: slow cross-confederation level. Treats each confederation as a
    single 'super-team' and solves the cross-confederation games over the
    recency-weighted multi-year window. Returns {confederation: level}.

    Pure function of games (no cache dependency). Cross-confed games are the
    only ones carrying between-confederation signal; World Cup games (tier
    2.0x) dominate, friendlies (0.25x) are present but muted. Returns {} when
    there is too little cross-confed evidence (early years) - callers then fall
    back to a zero prior (shrink toward global mean).
    """
    if len(cross_win) < 30:
        return {}
    confs = sorted(set(cross_win["home_confederation"]) | set(cross_win["away_confederation"]))
    cidx = {c: i for i, c in enumerate(confs)}
    n = len(confs)
    ng = len(cross_win)

    days_ago = (as_of_date - cross_win["date"]).dt.days.to_numpy(dtype=float)
    rec = 0.5 ** (days_ago / halflife_days)
    w = rec * cross_win["tier_weight"].to_numpy(float) * cross_win["match_type_weight"].to_numpy(float)
    hc = cross_win["home_confederation"].to_numpy()
    ac = cross_win["away_confederation"].to_numpy()
    adj = cross_win["adj_margin_home"].to_numpy(float)

    X = np.zeros((ng + 1, n)); y = np.zeros(ng + 1); ww = np.zeros(ng + 1)
    for k in range(ng):
        X[k, cidx[hc[k]]] =  1.0
        X[k, cidx[ac[k]]] = -1.0
    y[:ng] = adj; ww[:ng] = w
    X[-1, :] = 1.0; ww[-1] = 1.0e8

    sw = np.sqrt(ww)
    r, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return dict(zip(confs, r))


def _calibrated_solve(window_df, base_offset, cgp_lookup):
    """
    Layers 1+3 together: a fixed-K confederation<->team iterative calibration.

      c = slow confed offset (Layer 2)
      repeat K times:
          ranked = solve teams, each shrunk toward c          (Layer 1)
          c = alpha*base_offset + (1-alpha)*(solved bloc levels)   (Layer 3)
                (re-estimated each pass, but ANCHORED to the slow offset so the
                 loop converges near the slow level instead of chasing the
                 reactive window - which is what kills the bloc lurch)

    Returns the final iteration's ranked DataFrame (name, rating, rank).
    """
    confs = sorted({CONFEDERATION_MAP.get(t) for t in
                    set(window_df["home_team"]) | set(window_df["away_team"])} - {None})
    prior = {c: base_offset.get(c, 0.0) for c in confs}
    ranked = None
    for it in range(calibration_iters):
        ranked = _solve_wls(window_df, prior, confed_prior_lambda)
        if it < calibration_iters - 1:
            r = ranked.copy()
            r["conf"] = r["name"].map(CONFEDERATION_MAP)
            r["cgp"] = r["name"].map(cgp_lookup).fillna(0)
            lev = {}
            for c in confs:
                sub = r[(r["conf"] == c) & (r["cgp"] >= min_competitive_games)].nlargest(8, "rating")
                lev[c] = sub["rating"].mean() if len(sub) else np.nan
            blended = {
                c: (recenter_anchor_alpha * base_offset.get(c, 0.0)
                    + (1 - recenter_anchor_alpha) * lev[c]) if pd.notna(lev[c])
                   else base_offset.get(c, 0.0)
                for c in confs
            }
            m = np.mean(list(blended.values())) if blended else 0.0
            prior = {c: blended[c] - m for c in confs}   # demean: only relative level matters
    return ranked


# ============================================================
# STEP 1 - LOAD AND CLEAN THE RAW DATA
# ============================================================

# Team-name normalization: collapse source-side renames to one canonical name.
# The source (martj42) renamed "China PR" -> "China" wholesale (~2026-04). Left
# unmerged, the append-only db-union below preserves the orphaned "China PR"
# rows forever, duplicating the entire national team. Normalize at load so the
# dedup collapses them. Any future rename should be added here (the orphan guard
# in STEP 6b prints a WARNING when one appears).
TEAM_NAME_NORMALIZATION = {
    'China PR': 'China',
}

def _normalize_team_names(frame, cols):
    for c in cols:
        if c in frame.columns:
            frame[c] = frame[c].replace(TEAM_NAME_NORMALIZATION)
    return frame


def _fd_key():
    """football-data.org key from env, falling back to zidane/.env (shared)."""
    v = os.environ.get('FOOTBALL_DATA_KEY')
    if v:
        return v
    env_path = os.path.expanduser('~/code/fakeronjan/sports/zidane/.env')
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith('FOOTBALL_DATA_KEY='):
                    return line.split('=', 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def _fd_local_date(utc_iso, city, country):
    """football-data.org UTC kickoff -> local match day (pd.Timestamp @ midnight).

    Uses the venue city's timezone (WC_VENUE_TZ), falling back to the country's,
    so a 00:00-UTC US-night kickoff resolves to the correct previous local day.
    Returns None if the timezone is unknown or the timestamp won't parse, so the
    caller keeps martj42's date rather than guessing.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tzname = WC_VENUE_TZ.get(city) or _COUNTRY_TZ_FALLBACK.get(country)
    if not tzname:
        return None
    try:
        dt = datetime.fromisoformat(utc_iso.replace('Z', '+00:00')).astimezone(ZoneInfo(tzname))
        return pd.Timestamp(dt.date())
    except (ValueError, TypeError):
        return None


def _overlay_live_scores(frame):
    """Overlay football-data.org onto in-progress major-tournament rows.

    football-data.org is the AUTHORITY for the live major (the martj42 CC0 feed
    lags 1-2 days and occasionally logs a wrong score or a stale fixture date).
    For each FD fixture matched to a martj42 row (unordered team pair, within
    +/-1 day of the UTC kickoff, since martj42 dates by local match day) this:
      * rewrites the row's DATE to the LOCAL match day (FD kickoff converted via
        the venue timezone, WC_VENUE_TZ) - so a stale martj42 date is corrected
        while a 00:00-UTC US-night game stays on its true local day; and
      * fills OR overwrites the score from FD's full-time result, in the row's
        own home/away orientation.
    Only the active competition's rows in a recent window are touched; everything
    else (all history) is untouched here and locked by the append-only DB union.
    Any failure (no key, network, rate limit, API change) is swallowed - the
    daily build must never break over the overlay. Returns the mutated frame.
    """
    import json
    from urllib.request import Request, urlopen

    today = pd.Timestamp('today').normalize()
    lo = today - pd.Timedelta(days=FD_OVERLAY_LOOKBACK_DAYS)
    hi = today + pd.Timedelta(days=FD_OVERLAY_LOOKAHEAD_DAYS)
    w_lo, w_hi = lo - pd.Timedelta(days=1), hi + pd.Timedelta(days=1)
    blank = frame['home_score'].isna() | frame['away_score'].isna()

    for code, tournament in FD_LIVE_COMPETITIONS.items():
        # Gate: only call the API when this tournament has recent rows at all
        # (played or pending) - i.e. it's in progress. Off-season there are none,
        # so no request is made (a cancelled/free key never matters). Widened from
        # blanks-only so a wrong score on an already-played game can still be
        # corrected once the last blank is filled.
        active = (frame['tournament'] == tournament) & (frame['date'] >= lo) & (frame['date'] <= hi)
        if not active.any():
            continue

        key = _fd_key()
        if not key:
            print(f"  Live overlay: {int(active.sum())} recent {tournament} row(s) but no "
                  f"FOOTBALL_DATA_KEY - skipping (martj42 will backfill).")
            continue

        try:
            req = Request(f"{FD_BASE}/competitions/{code}/matches"
                          f"?dateFrom={w_lo.date()}&dateTo={w_hi.date()}",
                          headers={'X-Auth-Token': key})
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
            matches = payload.get('matches', [])
        except Exception as e:
            print(f"  Live overlay: football-data.org fetch failed for {code} "
                  f"({type(e).__name__}: {e}) - skipping, martj42 will backfill.")
            continue

        # Index every one of this tournament's rows in the (widened) window by
        # unordered team pair. `pair_seen` lets us tell a genuine gap (missing
        # fixture / bad alias) from a game martj42 already has, which is fine.
        window = (frame['tournament'] == tournament) & (frame['date'] >= w_lo) & (frame['date'] <= w_hi)
        rows_by_pair, pair_seen = {}, set()
        for i, r in frame.loc[window, ['date', 'home_team', 'away_team']].iterrows():
            pk = frozenset((r['home_team'], r['away_team']))
            pair_seen.add(pk)
            rows_by_pair.setdefault(pk, []).append(i)

        filled = corrected = redated = 0
        unmatched, used = [], set()
        for m in matches:
            fh = FD_TEAM_ALIASES.get(m['homeTeam']['name'], m['homeTeam']['name'])
            fa = FD_TEAM_ALIASES.get(m['awayTeam']['name'], m['awayTeam']['name'])
            utc = m['utcDate']
            fd_date = pd.Timestamp(utc[:10])
            # martj42's home/away score is the pre-shootout result (end of
            # regulation + extra time); the shootout winner lives in shootouts.csv
            # and drives the ±margin downstream. FD's `fullTime` FOLDS IN the
            # penalties for a shootout (e.g. 1-1 AET, 3-4 pens -> fullTime 4-5),
            # so for those we take regularTime+extraTime; otherwise fullTime is
            # already the AET/regulation result. A shootout row missing its
            # regularTime is left to martj42 rather than guessed.
            sc = m.get('score') or {}
            if sc.get('duration') == 'PENALTY_SHOOTOUT':
                rt, et = sc.get('regularTime') or {}, sc.get('extraTime') or {}
                if rt.get('home') is None or rt.get('away') is None:
                    hs = as_ = None
                else:
                    hs = rt['home'] + (et.get('home') or 0)
                    as_ = rt['away'] + (et.get('away') or 0)
            else:
                ft = sc.get('fullTime') or {}
                hs, as_ = ft.get('home'), ft.get('away')
            has_score = hs is not None and as_ is not None
            if fd_date < w_lo or fd_date > w_hi:
                continue
            pk = frozenset((fh, fa))

            # Closest unused row for this pair within +/-1 day of the UTC date.
            cand = sorted((i for i in rows_by_pair.get(pk, [])
                           if i not in used and abs((frame.at[i, 'date'] - fd_date).days) <= 1),
                          key=lambda i: abs((frame.at[i, 'date'] - fd_date).days))
            if not cand:
                # Genuine gap only for a scored game squarely inside the window
                # that martj42 has NO row for (missing fixture / bad alias).
                if has_score and pk not in pair_seen and lo <= fd_date <= hi:
                    unmatched.append(f"{fh} {hs}-{as_} {fa} ({fd_date.date()})")
                continue
            i = cand[0]
            used.add(i)

            # Date -> local match day from the venue timezone.
            ld = _fd_local_date(utc, frame.at[i, 'city'], frame.at[i, 'country'])
            if ld is not None and ld != frame.at[i, 'date']:
                frame.at[i, 'date'] = ld
                redated += 1

            # Score: fill a blank, or overwrite a martj42 value that disagrees.
            if has_score:
                nh, na = (hs, as_) if frame.at[i, 'home_team'] == fh else (as_, hs)
                was_blank = pd.isna(frame.at[i, 'home_score']) or pd.isna(frame.at[i, 'away_score'])
                if was_blank:
                    filled += 1
                elif frame.at[i, 'home_score'] != nh or frame.at[i, 'away_score'] != na:
                    corrected += 1
                frame.at[i, 'home_score'], frame.at[i, 'away_score'] = nh, na

        print(f"  Live overlay [{tournament}]: filled {filled}, overwrote {corrected}, "
              f"re-dated {redated} row(s) from football-data.org.")
        if unmatched:
            print(f"  Live overlay [{tournament}]: {len(unmatched)} FINISHED match(es) "
                  f"could NOT be placed (check FD_TEAM_ALIASES / dates): {unmatched}")
    return frame


# ── Manual source corrections (thin escape hatch) ───────────────────────────
# football-data.org is the authority for the live major and normally fixes
# martj42's wrong scores/dates automatically (see _overlay_live_scores). This
# list is the deterministic fallback for when the overlay CAN'T run (no key,
# outage) or for a known martj42 error outside FD's free-tier coverage. Each
# entry patches ONE row on its CURRENT (date, home, away) - and, for scores, the
# exact bad value - so it self-deactivates the moment martj42 (or FD) corrects
# upstream. Applied to BOTH the fresh download and the committed DB so a date
# rewrite lines the corrected rows up on the merge key instead of duplicating.
# Prune when martj42 has caught up.
SOURCE_CORRECTIONS = [
    # R16 date slips: FIFA schedules both on 2026-07-07; martj42 shows 07-06.
    # (FD's tz-aware overlay fixes these too; kept so an FD outage can't strand a
    # stale-dated pending twin as a duplicate.)
    {'date': '2026-07-06', 'home': 'Argentina',   'away': 'Egypt',    'new_date': '2026-07-07'},
    {'date': '2026-07-06', 'home': 'Switzerland', 'away': 'Colombia', 'new_date': '2026-07-07'},
]


def _apply_source_corrections(frame):
    """Patch known-bad martj42 rows in place (see SOURCE_CORRECTIONS)."""
    dates = pd.to_datetime(frame['date'])
    for c in SOURCE_CORRECTIONS:
        m = ((dates == pd.Timestamp(c['date']))
             & (frame['home_team'] == c['home']) & (frame['away_team'] == c['away']))
        if 'expect' in c:
            m &= (frame['home_score'] == c['expect'][0]) & (frame['away_score'] == c['expect'][1])
        if not m.any():
            continue
        if 'new_date' in c:
            frame.loc[m, 'date'] = pd.Timestamp(c['new_date'])
        if 'home_score' in c:
            frame.loc[m, 'home_score'] = c['home_score']
            frame.loc[m, 'away_score'] = c['away_score']
        chg = {k: v for k, v in c.items() if k in ('new_date', 'home_score', 'away_score')}
        print(f"  Source correction [{int(m.sum())} row]: "
              f"{c['home']} v {c['away']} ({c['date']}) -> {chg}")
    return frame


print("Loading results from GitHub...")
raw_df = pd.read_csv(data_url)
shootouts_df = pd.read_csv(shootouts_url)

# TBD fixture placeholders (e.g. an unset World Cup final date/venue with no
# teams assigned yet) show up as NaN home_team/away_team rows in the source -
# drop them before anything downstream assumes team names are strings.
_n_tbd = raw_df['home_team'].isna().sum() + raw_df['away_team'].isna().sum()
if _n_tbd:
    print(f"Dropping {_n_tbd} TBD fixture placeholder row(s) with no teams assigned.")
    raw_df = raw_df.dropna(subset=['home_team', 'away_team']).copy()

_normalize_team_names(raw_df, ['home_team', 'away_team'])
_normalize_team_names(shootouts_df, ['home_team', 'away_team', 'winner'])

print(f"Raw results loaded: {len(raw_df)} rows")
print(f"Shootouts loaded: {len(shootouts_df)} rows")

raw_df['date'] = pd.to_datetime(raw_df['date'])
shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])

# Deterministic fallback corrections (see SOURCE_CORRECTIONS) before the FD
# overlay and DB union. The overlay normally supersedes these; they matter when
# it can't run, and keep a stale-dated pending row from duplicating.
raw_df = _apply_source_corrections(raw_df)

# Best-effort live-score overlay for an in-progress major (see config above).
# Wrapped so a bug or outage here can never fail the daily ratings build.
try:
    raw_df = _overlay_live_scores(raw_df)
except Exception as e:
    print(f"  Live overlay skipped ({type(e).__name__}: {e}) - using martj42 as-is.")

# Date filter + drop U-23 events. Everything else (competitive + friendlies)
# stays in the dataset - friendlies will get downweighted via WLS observation
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
# adjustment - WC finals, CFP-style neutrals, international kickoffs.

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

# Shootout overrides - winner gets full W, loser full L (for standings only,
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
# As-of-date override for teams that switched confederation (Australia, etc.),
# so cross-confederation calibration uses the confederation in effect then.
_apply_confederation_history(df, 'home_team', 'home_confederation', 'date')
_apply_confederation_history(df, 'away_team', 'away_confederation', 'date')

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

# Append-only DB (DUNCAN model): all_soccer_games.csv is the source of truth.
# martj42 is re-fetched in full every run, but it does NOT get to re-derive the
# whole database. Games older than DB_LOCK_DAYS are LOCKED to the committed file
# - the re-fetch can only ADD new history, never overwrite a recorded result (so
# a flaky/short fetch or a late upstream edit can't silently corrupt or shift the
# positional date-id cache). Only the "active zone" (recent + near-future:
# in-progress tournaments, games martj42 is still backfilling) stays fresh-
# authoritative, so pending games finalize and live/FD corrections land. Games
# the fetch didn't return at all are always preserved. History grows and the
# active tail is corrected; settled history never silently shrinks or churns.
if os.path.exists('all_soccer_games.csv'):
    _prev = pd.read_csv('all_soccer_games.csv')
    _prev['date'] = pd.to_datetime(_prev['date'], errors='coerce')
    _prev = _prev.drop(columns=[c for c in ('grouped_date_id', 'unique_game_id') if c in _prev.columns])
    # Same corrections on the committed DB, so a date rewrite lines the row up
    # with its corrected fresh twin on the merge key instead of duplicating.
    _prev = _apply_source_corrections(_prev)
    # Normalize committed names too, so a source rename collapses on the dedup
    # below instead of leaving an orphaned duplicate of the renamed team.
    _normalize_team_names(_prev, ['home_team', 'away_team', 'shootout_winner'])

    # Orphan-rename guard: any team in the committed DB that the live source no
    # longer uses is a likely source rename - it will be permanently duplicated
    # by the preservation logic below unless added to TEAM_NAME_NORMALIZATION.
    _src_teams = set(raw_df['home_team']) | set(raw_df['away_team'])
    _orphans = (set(_prev['home_team']) | set(_prev['away_team'])) - _src_teams - {np.nan}
    if _orphans:
        print(f"WARNING: {len(_orphans)} team(s) in committed DB absent from live source "
              f"(possible source rename - add to TEAM_NAME_NORMALIZATION): {sorted(str(t) for t in _orphans)}")

    _key = ['date', 'home_team', 'away_team']
    _horizon = pd.Timestamp('today').normalize() - pd.Timedelta(days=DB_LOCK_DAYS)
    # Per-row merge priority (lower wins the dedup). Active zone (date >= horizon):
    # fresh is authoritative (0) so it finalizes pending games and lands live/FD
    # corrections. Locked history (date < horizon): the committed DB wins (0) and
    # the re-fetch (1) can only contribute rows for keys the DB doesn't have yet.
    _fresh = df.copy()
    _prevp = _prev.copy()
    _fresh['_pri'] = np.where(_fresh['date'] >= _horizon, 0, 1)
    _prevp['_pri'] = np.where(_prevp['date'] >= _horizon, 1, 0)
    _combined = pd.concat([_fresh, _prevp], ignore_index=True, sort=False)
    _combined = _combined.sort_values('_pri', kind='mergesort').drop_duplicates(subset=_key, keep='first')
    _fk = set(map(tuple, df[_key].astype(str).values))
    _pk = set(map(tuple, _prev[_key].astype(str).values))
    _added, _preserved = len(_fk - _pk), len(_pk - _fk)
    _locked = int((_prevp['date'] < _horizon).sum())
    print(f"[db-union] {len(df):,} fresh + {len(_prev):,} committed -> {len(_combined):,} rows "
          f"(+{_added:,} new, {_preserved:,} preserved from DB, "
          f"~{_locked:,} historical rows locked to committed values; horizon {_horizon.date()})")
    df = _combined.drop(columns=['_pri']).reset_index(drop=True)

# ============================================================
# STEP 7 - DATE IDs FOR ROLLING WINDOW
# ============================================================

# Canonical, stable row order (date, then teams) so the committed DB is a
# deterministic snapshot: the append-only union feeds fresh + locked rows in a
# merge-dependent order, and a plain date sort leaves within-date ties to
# reshuffle each run (huge spurious diffs). Sorting on the full game key pins the
# order to content, so a run only rewrites rows whose data actually changed.
df.sort_values(['date', 'home_team', 'away_team'], kind='stable', inplace=True)
df.reset_index(drop=True, inplace=True)

df['grouped_date_id'] = df.groupby('date').ngroup() + 1
df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

df.to_csv('all_soccer_games.csv', index=False)
print("Master game CSV saved: all_soccer_games.csv")

# ============================================================
# STEP 7b - LAST MATCH STRINGS
# ============================================================
# Format: "W vs. France 2-1 (UEFA Euro)" or "L @ Brazil 0-3 (Friendly)"
# Includes friendlies - fans want to see the most recent game played
# regardless of competitiveness.

df['home_score_int'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score_int'] = pd.to_numeric(df['away_score'], errors='coerce')

# In-progress editions: any (tournament, year) that still has games scheduled but
# not yet played (NaN score, dated today or later) is ONGOING and must not be
# crowned a podium. The source schedule carries future fixtures, so we capture
# this here, BEFORE the dropna strips unplayed rows. Without it the podium
# bracket-walk can spuriously declare a "final" mid-tournament (e.g. crowning a
# World Cup champion during the group stage).
_today_ts = pd.Timestamp(date.today())
_unplayed = df[
    (df['home_score_int'].isna() | df['away_score_int'].isna())
    & (df['date'] >= _today_ts)
]
in_progress_editions = {
    (t, int(y)) for t, y in zip(_unplayed['tournament'], _unplayed['date'].dt.year)
}
if in_progress_editions:
    print(f"In-progress editions (podium suppressed): {sorted(in_progress_editions)}")

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
    df['home_result_flag'] + ' ' +
    df['home_score_int'].map(str) + '-' + df['away_score_int'].map(str) +
    df['home_venue'] +
    df['away_team'] +
    ' (' + df['tournament'] + ')'
)
df['away_last_match'] = (
    df['away_result_flag'] + ' ' +
    df['away_score_int'].map(str) + '-' + df['home_score_int'].map(str) +
    df['away_venue'] +
    df['home_team'] +
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
# One snapshot per game-day; each uses a rolling window_years calendar window.
# WLS observation weight per game = recency × tournament tier × match type.
# Incremental: skips date IDs already in messi_ratings.csv.gz cache.
# Cache is gzipped per the soccer-family convention (would exceed
# GitHub's 50MB recommendation uncompressed).

print("Starting MESSI rating calculations...")

max_date_id = int(df['grouped_date_id'].max())
min_date_id = 1

try:
    messi_df = pd.read_csv('messi_ratings.csv.gz')
except FileNotFoundError:
    messi_df = None
    print("No existing ratings found - running full history from scratch.")

# Cache-validity guard: ranking_id is positional (groupby('date').ngroup()+1),
# so if the game-date set changes size (e.g. the re-fetched results set shifts),
# cached ids desync from dates and the skip logic freezes ratings at an old
# date. Verify the cached id->date mapping still matches current games; on any
# mismatch, discard the cache and rebuild from scratch. (Same guard as COBI/
# ZIDANE; the db-union above should keep the set stable, this is the backstop.)
if messi_df is not None:
    cur_id_date = (df.drop_duplicates('grouped_date_id')
                     .set_index('grouped_date_id')['date']
                     .dt.strftime('%Y-%m-%d').to_dict())
    cache_id_date = (messi_df.drop_duplicates('ranking_id')
                       .set_index('ranking_id')['ranking_date']
                       .astype(str).str.slice(0, 10).to_dict())
    mismatches = sum(1 for rid, d in cache_id_date.items() if cur_id_date.get(rid) != d)
    if mismatches:
        print(f"  cache desynced from current game dates "
              f"({mismatches:,} ranking_id<->date mismatches) - full rebuild from scratch")
        messi_df = None

if messi_df is None:
    messi_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_id_ranked = -1
    min_id_ranked = -1
else:
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

last_printed_ym = None

# Cross-confederation games only carry between-confederation signal; pre-slice
# once. The slow confed offset (Layer 2) is recomputed once per calendar month
# (it barely moves day-to-day over an 8yr window) and reused within the month.
cross_all = df[df['home_confederation'] != df['away_confederation']].copy()
_offset = {}
_offset_ym = None

for i in range(min_date_id, max_date_id + 1):

    if min_id_ranked <= i <= max_id_ranked:
        continue

    current_date = df.loc[df['grouped_date_id'] == i, 'date'].max()
    if pd.isnull(current_date):
        continue

    # Calendar-year window: all games within the last window_years of this
    # game-day. Snapshot cadence stays one-per-game-day (ranking_id = i).
    window_start = current_date - pd.DateOffset(years=window_years)
    working_df = df.loc[
        (df['date'] > window_start) &
        (df['date'] <= current_date)
    ].copy()

    if len(working_df) < 10:
        continue

    # Combined WLS observation weight: recency × tier × match_type.
    # Recency is an exponential half-life in CALENDAR days (decoupled from the
    # window length): recent form dominates, older pedigree lingers lightly.
    working_df['days_ago'] = (current_date - working_df['date']).dt.days
    working_df['date_weight'] = 0.5 ** (working_df['days_ago'] / (recency_halflife_years * 365.0))
    working_df['weight'] = (
        working_df['date_weight'] *
        working_df['tier_weight'] *
        working_df['match_type_weight']
    )

    # Drop zero-weight rows only. We deliberately KEEP 0-margin draws: a draw
    # between mismatched teams is strong signal (the WLS row rating_home -
    # rating_away = adj_margin pulls them together). The old `adj_margin_home
    # != 0` filter was a vestigial carryover from the pre-port rankit pipeline
    # ("the solver itself doesn't need this") and it silently dropped every
    # NEUTRAL-site regulation draw -- because neutral games get hfa=0, so a
    # neutral draw is the only case where adj_margin_home lands exactly on 0.
    # That ate ALL World Cup draws (WC venues are neutral), e.g. Spain 0-0 Cape
    # Verde 2026-06-15 never entered the solve. Fixed 2026-06-18.
    working_df = working_df[working_df['weight'] > 0]
    if len(working_df) < 10:
        continue

    season = current_date.year
    current_ym = current_date.strftime('%Y-%m')
    if current_ym != last_printed_ym:
        pct = round(100 * i / max_date_id)
        print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct}% complete)")
        last_printed_ym = current_ym

    try:
        # Competitive (non-friendly) game counts: feed both the calibration
        # bloc-means (Layer 3) and the final eligibility filter.
        comp_df = working_df[working_df['match_type'] == 'competitive']
        cgp_lookup = pd.concat([comp_df['home_team'], comp_df['away_team']]).value_counts()

        # Layer 2: refresh the slow confederation offset once per calendar month.
        if current_ym != _offset_ym:
            cross_win = cross_all[
                (cross_all['date'] > current_date - pd.Timedelta(days=365 * confed_offset_years)) &
                (cross_all['date'] <= current_date)
            ]
            _offset = _confed_offset(cross_win, current_date, confed_offset_halflife_d)
            _offset_ym = current_ym

        # Layers 1+3: fixed-K confederation<->team iterative calibration.
        ranked = _calibrated_solve(working_df, _offset, cgp_lookup)
        if ranked['rating'].isna().any() or np.isinf(ranked['rating']).any():
            continue

        # Offense/defense split on the same window, re-anchored so that
        # rating_o + rating_d == the calibrated rating. delta is per team:
        # adding it to both sides preserves the attack-vs-defense lean
        # (rating_o - rating_d) from the raw solve while pinning the level to
        # the calibrated rating, so the cap / shootout / pooling treatment
        # flows through. sum(delta) = 0, so both vectors stay zero-sum.
        ranked_od = _solve_wls_od(working_df, off_share=od_off_share, goal_cap=od_goal_cap)
        ranked = ranked.merge(ranked_od, on='name', how='left')
        delta = (ranked['rating'] - ranked['rating_o'] - ranked['rating_d']) / 2.0
        ranked['rating_o'] = ranked['rating_o'] + delta
        ranked['rating_d'] = ranked['rating_d'] + delta
        ranked['rank_o'] = ranked['rating_o'].rank(ascending=False, method='min').astype(int)
        ranked['rank_d'] = ranked['rating_d'].rank(ascending=False, method='min').astype(int)

        ranked['ranking_id']   = i
        ranked['ranking_date'] = current_date.date()
        ranked['season']       = season

        # Total games played (all match types) for output/display.
        gp_tot = working_df.groupby('home_team').size().add(
                 working_df.groupby('away_team').size(), fill_value=0)
        ranked['games_played'] = ranked['name'].map(gp_tot).fillna(0).astype(int)
        ranked['competitive_games_played'] = ranked['name'].map(cgp_lookup).fillna(0).astype(int)

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

    # Don't crown a champion for a tournament that isn't over:
    #   (a) it still has loaded fixtures left to play (group stage in progress), or
    #   (b) it has a known final date we haven't reached yet (knockout fixtures
    #       aren't in the feed until teams are set, so "no future games loaded"
    #       does NOT mean the edition is finished).
    if (tournament, int(year)) in in_progress_editions:
        continue
    _end_date = TOURNAMENT_END_DATES.get((tournament, int(year)))
    if _end_date is not None and date.today() < _end_date:
        continue

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
        # Bracket-walk couldn't find a clean semifinal->final structure. With a
        # SINGLE game on the last date that game is the decider - safe to crown
        # (two-legged Copa finals, Oceania/Euro one-off finals). With MULTIPLE
        # last-day games it's genuinely ambiguous - a group/league-phase finish
        # with no final at all (CONCACAF Nations League 2019 -> Puerto Rico;
        # African Cup 2025 Dec group games; UEFA Nations League league phase),
        # or the real final sitting beside a lower-stakes game (1995 King Fahd
        # Cup: Denmark 2-0 Argentina AND Mexico 1-1 Nigeria same day). Picking
        # arbitrarily crowned the wrong side every time, so don't guess.
        if len(final_games) > 1:
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
# Same for the 'name' merge_asof key. On a FULL rebuild messi_df is seeded from
# an empty pd.DataFrame(columns=[...]) whose columns are object dtype, so
# concatenating the str-dtype solver frames onto it downcasts 'name' to object.
# lastmatch_df['name'] is str (read_csv under pandas-3 infer_string), and
# merge_asof requires the 'by' keys to match exactly - so normalize to str.
# (Incremental runs read messi_df from CSV, so 'name' is already str and match.)
final_df['name'] = final_df['name'].astype(str)
final_df['date'] = pd.to_datetime(final_df['date'])

# Confederation (as-of snapshot date, honoring historical switches)
final_df['confederation'] = final_df['name'].map(CONFEDERATION_MAP).fillna('Unknown')
_apply_confederation_history(final_df, 'name', 'confederation', 'date')

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
    'rating_o', 'rank_o', 'rating_d', 'rank_d',
    'games_played', 'competitive_games_played',
    'last_match_date', 'last_match', 'is_game_day',
    'most_recent', 'is_world_cup_final_day',
    'tournament_finish'
]]

final_df.sort_values(['ranking_id', 'rank'], inplace=True)
final_df.drop_duplicates(keep='first', inplace=True)

# 1986+ cutoff (data quality cliff before - Maradona-era WC anchor)
final_df = final_df[final_df['date'] >= pd.to_datetime('1986-01-01').date()]

# Eligibility filter: require min_competitive_games NON-FRIENDLY games in window.
# Friendlies still contribute to the regression (cross-confederation signal)
# but don't count toward eligibility. This fixes the Israel-1998 hot-streak
# anomaly where 5 friendlies inflated a team to #1.
final_df = final_df[final_df['competitive_games_played'] >= min_competitive_games]
print(f"After min_competitive_games={min_competitive_games} filter: {len(final_df)} rows")

# Re-rank within each snapshot AFTER the eligibility filter. rank / rank_o /
# rank_d are first assigned over the full window (in the ratings loop), which
# includes sub-threshold teams that never appear in the displayed table. That
# was invisible for the net rank (a no-sample minnow sits near the bottom, so
# its gap is buried deep in the list) but showed on the O/D dimensions, where a
# 0-competitive-game side can carry an artifact-high rating_d and occupy "D#1" -
# leaving the world's best real defense labeled D#2 with no visible D#1, and
# every eligible team's O/D rank silently offset. Ranking among eligible teams
# only makes all three contiguous and correct. (method='min' matches upstream.)
for _rank_col, _rating_col in (('rank', 'rating'), ('rank_o', 'rating_o'), ('rank_d', 'rating_d')):
    final_df[_rank_col] = (final_df.groupby('ranking_id')[_rating_col]
                                   .rank(ascending=False, method='min').astype(int))

final_df.to_csv('messi_ratings_final.csv.gz', index=False, compression='gzip')
print("messi_ratings_final.csv.gz saved!")
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
