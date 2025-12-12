"""
Check TB vs ATL game status in data
"""

import pandas as pd

df = pd.read_csv('nfl_training_data_ultimate.csv')

print("="*70)
print("CHECKING TB vs ATL GAME")
print("="*70)

# Current season
current_season = df['season'].max()
print(f"\nCurrent Season: {current_season}")

# Find TB vs ATL game
tb_atl = df[
    ((df['home_team'] == 'TB') & (df['away_team'] == 'ATL')) |
    ((df['home_team'] == 'ATL') & (df['away_team'] == 'TB'))
].copy()

tb_atl = tb_atl[tb_atl['season'] == current_season]

print(f"\nTB vs ATL games in {current_season}:")
print("─"*70)

for idx, game in tb_atl.iterrows():
    print(f"\nWeek {game['week']}: {game['away_team']} @ {game['home_team']}")
    print(f"  Home Score: {game.get('home_score', 'NaN')}")
    print(f"  Away Score: {game.get('away_score', 'NaN')}")
    print(f"  Date: {game.get('gameday', 'Unknown')}")
    
    if pd.notna(game.get('home_score')):
        print(f"  ✅ COMPLETED")
    else:
        print(f"  ⏳ NOT PLAYED YET (in data)")

# Check Week 15 specifically
print("\n" + "="*70)
print("WEEK 15 GAMES STATUS")
print("="*70)

week_15 = df[(df['season'] == current_season) & (df['week'] == 15)].copy()

completed = week_15[week_15['home_score'].notna()]
upcoming = week_15[week_15['home_score'].isna()]

print(f"\nTotal Week 15 games: {len(week_15)}")
print(f"  Completed: {len(completed)}")
print(f"  Upcoming: {len(upcoming)}")

if len(upcoming) > 0:
    print(f"\n⏳ Upcoming games:")
    for idx, game in upcoming.iterrows():
        print(f"  • {game['away_team']} @ {game['home_team']}")

if len(completed) > 0:
    print(f"\n✅ Completed games:")
    for idx, game in completed.iterrows():
        print(f"  • {game['away_team']} @ {game['home_team']}: {game['away_score']:.0f}-{game['home_score']:.0f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if len(upcoming) == 0:
    print("\n⚠️  ALL Week 15 games are completed in the data!")
    print("   Your prediction script will show Week 15 as 'last completed week'")
    print("\n   OPTIONS:")
    print("   1. Re-run 01_data_collection.py to get latest data")
    print("   2. Or script will show completed games (with actual scores)")
else:
    print(f"\n✅ {len(upcoming)} games still upcoming")
    print("   TB vs ATL should NOT be in predictions if it's completed")
    print("\n   If TB vs ATL is showing, the data hasn't been updated yet")
    print("   Re-run 01_data_collection.py to fetch latest results")

print()