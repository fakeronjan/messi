"""
3-way comparison of MESSI1 (competitive), MESSI2 (all), MESSI3 (WC only).
For each model, show top 25 (country, year) at tournament-end snapshots.
"""
import pandas as pd

df = pd.read_csv('messi_ratings_final.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Load games for tournament-end dates
games = pd.read_csv('all_soccer_games.csv')
games['date'] = pd.to_datetime(games['date'])
PODIUM_TOURNAMENTS = [
    'FIFA World Cup', 'UEFA Euro', 'Copa América', 'AFC Asian Cup',
    'UEFA Nations League', 'CONCACAF Nations League',
]
final_dates = set()
for t in PODIUM_TOURNAMENTS:
    tg = games[games['tournament'] == t]
    if tg.empty:
        continue
    for year, grp in tg.groupby(tg['date'].dt.year):
        final_dates.add(grp['date'].max().date())

eos = df[df['date'].dt.date.isin(final_dates)].copy()
print(f"Tournament-end snapshots in data: {len(eos)}")

def top25(col, label):
    print(f"\n{label} top 25:")
    sub = eos.dropna(subset=[col])
    sub = sub.sort_values(col, ascending=False).drop_duplicates(['country', 'year'], keep='first').head(25)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"  #{i:2}  {r['country']:<16} {int(r['year'])}  {r[col]:>6.2f}")

top25('rating',  'MESSI1 (competitive only)')
top25('rating2', 'MESSI2 (all matches incl friendlies)')
top25('rating3', 'MESSI3 (WC qualifying + finals only)')
top25('rating_blend', 'BLEND (current formula)')
