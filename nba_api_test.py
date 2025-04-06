import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelogs
import time
from tqdm import tqdm

# Define seasons
seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2000, 2024)]

# Get all NBA teams
nba_teams = teams.get_teams()

# Master list to store data
all_data = []

for season in tqdm(seasons, desc="Fetching data by season"):
    for team in nba_teams:
        team_id = team['id']
        try:
            logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
            df = logs.get_data_frames()[0]
            df['SEASON'] = season
            all_data.append(df)
            time.sleep(0.6)  # Prevent rate-limiting
        except Exception as e:
            print(f"Failed: {team['full_name']} ({season}) | Error: {e}")
            continue

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)

# Save to CSV
full_df.to_csv("nba_team_game_logs_2000_2024.csv", index=False)
print("✅ Saved to nba_team_game_logs_2000_2024.csv")
