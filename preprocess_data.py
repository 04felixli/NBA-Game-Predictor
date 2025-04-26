import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelogs
import time
from tqdm import tqdm
from collections import defaultdict


def fetch_team_game_logs(season):
    """
    Fetch NBA team game logs for a given season.
    Args:
        season (str): The NBA season in the format "YYYY-YY" (to work with the nba_api).
    Returns:
        pd.DataFrame: A DataFrame containing game logs for all teams in the specified season.
    """
    # Get all NBA teams using nba_api library
    nba_teams = teams.get_teams()

    # Master list to store data (a list of DataFrames)
    all_data = []

    for team in tqdm(nba_teams, desc=f"Fetching data for season {season}"):
        team_id = team['id']
        try:
            logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
            df = logs.get_data_frames()[0]
            df['SEASON'] = season # add season column
            all_data.append(df)
            time.sleep(0.6)  # To avoid hitting the API too hard
        except Exception as e:
            print(f"Failed: {team['full_name']} ({season}) | Error: {e}")
            continue

    return pd.concat(all_data, ignore_index=True)

def compute_recent_performance(df):
    """
    Compute recent performance metrics for each team over the past 10 games.
    If a team has played less than 10 games, it will compute the average of available games.
    Args:
        df (pd.DataFrame): DataFrame containing all game logs for all teams over multiple seasons.
    Returns:
        pd.DataFrame: DataFrame with additional columns for recent performance metrics.
    """
    # Sort games by date so that each team's games are in order for every season
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'SEASON', 'GAME_DATE'])

    # Stats to average
    stats_to_avg = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV', 'STL', 'BLK']

    # Compute rolling averages for each stat
    for stat in stats_to_avg:
        rolling_stat = (
            df.groupby(['TEAM_ID', 'SEASON'])[stat]
              .apply(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
              .reset_index(drop=True)
        )

        # Combine: use rolling average if exists
        df[f'{stat}_avg10'] = rolling_stat

    # Drop first game of each team-season pair since there are no previous games
    df = df.dropna(subset=[f'{stat}_avg10' for stat in stats_to_avg])
        
    return df

def compute_elo(df, base_elo=1500, k=20): 
    """
    Compute Elo ratings for each team based on game results.
    Args:
        df (pd.DataFrame): DataFrame containing game logs with game results.
        base_elo (int): Initial Elo rating for all teams.
        k (int): K-factor for Elo rating calculation.
    Returns:
        pd.DataFrame: DataFrame with additional columns for Elo ratings.
    """   
    # Sort games by date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')

    # Initialize team Elo ratings
    team_elos = defaultdict(lambda: base_elo)

    # Store Elo values for each team in each game
    elo_records = []

    # Ensure home/away flags and game key exist
    if 'game_key' not in df.columns:
        df['game_key'] = df['GAME_ID']
    if 'is_home' not in df.columns:
        df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    for game_id in df['GAME_ID'].unique():
        game = df[df['GAME_ID'] == game_id]
        if len(game) != 2:
            continue  # Skip if both teams not found

        team_home = game[game['is_home'] == 1]
        team_away = game[game['is_home'] == 0]

        if team_home.empty or team_away.empty:
            continue

        # Extract team IDs
        home_id = team_home.iloc[0]['TEAM_ID']
        away_id = team_away.iloc[0]['TEAM_ID']

        # Current Elo
        home_elo = team_elos[home_id]
        away_elo = team_elos[away_id]

        # Who won?
        home_won = 1 if team_home.iloc[0]['WL'] == 'W' else 0

        # Expected scores
        # Apply home court adjustment for expected score only
        home_court_advantage = 100
        home_elo_adj = home_elo + home_court_advantage
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo_adj) / 400))
        expected_away = 1 - expected_home

        # Update Elos
        new_home_elo = home_elo + k * (home_won - expected_home)
        new_away_elo = away_elo + k * ((1 - home_won) - expected_away)

        # Save current Elo values to the game rows
        elo_records.append({'GAME_ID': game_id, 'TEAM_ID': home_id, 'ELO': home_elo})
        elo_records.append({'GAME_ID': game_id, 'TEAM_ID': away_id, 'ELO': away_elo})

        # Update Elo history
        team_elos[home_id] = new_home_elo
        team_elos[away_id] = new_away_elo

    # Merge back into the original DataFrame
    elo_df = pd.DataFrame(elo_records)
    df = pd.merge(df, elo_df, on=['GAME_ID', 'TEAM_ID'], how='left')

    return df


def main():
    fetch_data = False

    if fetch_data:
        # Define seasons: 2015-16 to 2023-24 for now. 
        seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2015, 2024)]

        # Master list to store all data
        all_data = []

        for season in seasons:
            df = fetch_team_game_logs(season)
            all_data.append(df)

        # Combine all data
        full_df = pd.concat(all_data, ignore_index=True)

        # Save to CSV
        full_df.to_csv("nba_team_game_logs_2015_2024.csv", index=False)
        print("Saved to nba_team_game_logs_2015_2024.csv")

    # assuming we have the data already saved
    else:
        # Load the data
        full_df = pd.read_csv("nba_team_game_logs_2015_2024.csv")
    
    # compute recent performance metrics
    full_df = compute_recent_performance(full_df)

    # compute Elo ratings
    full_df = compute_elo(full_df)

    # Save the processed data
    full_df.to_csv("nba_team_game_logs_processed.csv", index=False)
    print("Processed data saved to nba_team_game_logs_processed.csv")

    # Add game key and home/away flag
    full_df['game_key'] = full_df['GAME_ID']
    full_df['is_home'] = full_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # Split into home and away DataFrames
    home_df = full_df[full_df['is_home'] == 1].copy()
    away_df = full_df[full_df['is_home'] == 0].copy()

    # Merge into matchup rows
    matchups = pd.merge(
        home_df,
        away_df,
        on='game_key',
        suffixes=('_home', '_away')
    )

    # Create target column
    matchups['home_win'] = (matchups['PTS_home'] > matchups['PTS_away']).astype(int)

    # Save matchups
    matchups.to_csv("matchups.csv", index=False)
    print("Matchups saved to matchups.csv")

    # Filter out unnecessary columns
    columns_to_keep = [
        # Elo
        'ELO_home', 'ELO_away',

        # Rolling averages (10 games)
        'PTS_avg10_home', 'REB_avg10_home', 'AST_avg10_home',
        'FG_PCT_avg10_home', 'FG3_PCT_avg10_home', 'FT_PCT_avg10_home',
        'TOV_avg10_home', 'STL_avg10_home', 'BLK_avg10_home',

        'PTS_avg10_away', 'REB_avg10_away', 'AST_avg10_away',
        'FG_PCT_avg10_away', 'FG3_PCT_avg10_away', 'FT_PCT_avg10_away',
        'TOV_avg10_away', 'STL_avg10_away', 'BLK_avg10_away',

        # Target
        'home_win'
    ]

    matchups_clean = matchups[columns_to_keep]
    matchups_clean.to_csv("training_dataset.csv", index=False)




if __name__ == "__main__":
    main()
