import argparse
import pandas as pd
import numpy as np

from features_config import PREPROCESSING_FEATURES

def preprocess_data(input_path, output_path):
    # Load the CSV file
    data = pd.read_csv(input_path)

    # Perform data preprocessing steps
    # Modify the code below to suit your specific data processing requirements
    processed_data = data.drop_duplicates()  # Example: Remove duplicates

    # Save the processed data to a new CSV file
    processed_data.to_csv(output_path, index=False)
    print("Data preprocessing completed. Processed CSV file saved at", output_path)


def create_games_df(input_path: str, remove_playoffs: bool = True):
    """
    Loads the data and processes it into a dataframe with the following columns
    GameId: unique identifier for each game
    TimeElapsed: time elapsed in seconds
    Home: home indicator
    Team: team of player
    Opponent: opponent of team
    TeamRest: days rest for team
    OpponentTeamRest: days rest for opponent
    """
    df = pd.read_csv(input_path)
    df['GameId'] = df.groupby(['HomeTeam', 'AwayTeam', 'Date']).ngroup()
    # Time Elapsed
    df['TimeElapsed'] = (np.where(df['Quarter']<=4, 720, 300) - df['SecLeft'] + 
                         (df['Quarter'].clip(1,4)-1)*720 + (df['Quarter']-5).clip(0)*300)
    df['TimeRemaining'] = df['SecLeft'] + (4 - df['Quarter'].clip(1,4))*720
    # Home Indicator
    df['Home'] = np.where(df['HomePlay'].isna(), 0, 1)
    # Days rest by team for opponent
    df['Date'] = pd.to_datetime(df['Date'])
    df['Team'] = np.where(df['HomePlay'].isna(), df['AwayTeam'], df['HomeTeam'])
    team_rest = df.groupby(['GameId', 'Team']).first().sort_values('Date') \
                  .groupby('Team')['Date'].diff().dt.days.fillna(0).clip(0,4).reset_index()
    df = df.merge(team_rest.rename(columns={'Date': 'TeamRest'}), on=['GameId', 'Team'])
    df['Opponent'] = np.where(df['HomePlay'].isna(), df['AwayTeam'], df['HomeTeam'])
    df = df.merge(team_rest.rename(columns={'Date': 'OpponentTeamRest'}), 
                  left_on=['GameId', 'Opponent'], right_on=['GameId', 'Team'], suffixes=('', '_y'))
    # Remove playoffs if indicated
    if remove_playoffs:
        df = df.loc[df['GameType'] == "regular"]

    return df

def interval_metric(df: pd.DataFrame, intervals: int, metric: str):
    """
    Calculates the metric per interval time in game for each player
    Results in dataframe with columns:
    Player, GameId, Cumulative{metric}_{i}, where i is the interval
    Extra columns for context as well
    """
    # Check points every 4 minutes
    for i in range(intervals):
        # Don't remove overtime points
        max_time = (48/intervals) * (i+1) * 60
        if (i == intervals-1):
            max_time = np.float('inf')
        cum_pts = df.groupby(['Player', 'GameId']) \
                    .apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, 
                            f'CumulativePlayer{metric}'].max()).reset_index().fillna(0)
        tot_pts = df.groupby(['GameId']) \
                    .apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, 
                            f'CumulativeTotal{metric}'].max()).reset_index().fillna(0)
        team_pts = df.groupby(['Team', 'GameId']) \
                     .apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, 
                            f'CumulativeTeam{metric}'].max()).reset_index().fillna(0)
        opponent_pts = df.groupby(['Team', 'GameId']) \
                         .apply(lambda x: x.loc[x['TimeElapsed'] <= max_time,
                                f'CumulativeOpponent{metric}'].max()).reset_index().fillna(0)
        df = df.merge(cum_pts.rename(columns={0: f'CumulativePlayer{metric}_{i}'}), on=['Player', 'GameId'], how='left')
        df = df.merge(tot_pts.rename(columns={0: f'CumulativeTotal{metric}_{i}'}), on=['GameId'], how='left')
        df = df.merge(team_pts.rename(columns={0: f'CumulativeTeam{metric}_{i}'}), on=['Team', 'GameId'], how='left')
        df = df.merge(opponent_pts.rename(columns={0: f'CumulativeOpponent{metric}_{i}'}), on=['Team', 'GameId'], how='left')

    return df

def add_player_game_scoring(df: pd.DataFrame, intervals: int = 4):
    """
    Calculate the cumulative points scored by each player in each game
    """
    scoring_play = ~((df['FreeThrowOutcome'].isna()) & (df['ShotOutcome'].isna()))
    df = df.loc[scoring_play]
    df['Player'] = np.where(~df['FreeThrowOutcome'].isna(), df['FreeThrowShooter'], df['Shooter'])
    df['PotentialPoints'] = np.where(df['ShotType'].str.startswith('3'), 3, 2)
    df['PotentialPoints'] = np.where(df['FreeThrowOutcome'].isna(), df['PotentialPoints'], 1)
    df['Points'] = ((df['FreeThrowOutcome'] =="make") | (df['ShotOutcome'] =="make")) * df['PotentialPoints']
    df['CumulativePlayerPoints'] = df.sort_values('TimeElapsed').groupby(['Player', 'GameId'])['Points'].cumsum()
    df['CumulativeTotalPoints'] = df.sort_values('TimeElapsed').groupby(['GameId'])['Points'].cumsum()
    df['CumulativeTeamPoints'] = df.sort_values('TimeElapsed').groupby(['Team', 'GameId'])['Points'].cumsum()
    df['CumulativeOpponentPoints'] =  df['CumulativeTotalPoints'] - df['CumulativeTeamPoints']
    df = interval_metric(df, intervals, 'Points')

    return df

def add_player_game_rolling_avg(df: pd.DataFrame, metric: str):
    """
    Calculates the rolling EWM average for each player in each game.
    Must have columns:
    Player, GameId, Cumulative{metric}
    """
    df = df.sort_values('Date')
    game_stats = df.groupby(['Player', 'GameId', 'Date'])[f'CumulativePlayer{metric}'].max().reset_index()
    game_stats[f'RollingAvgPlayer{metric}'] = game_stats.sort_values('Date') \
                                                        .groupby(['Player'])[f'CumulativePlayer{metric}'] \
                                                        .transform(lambda x: x.ewm(span=5).mean())
    game_stats["PlayerGameNumber"] = game_stats.groupby(['Player']).cumcount()
    df = df.merge(game_stats[['Player', 'GameId', f'RollingAvgPlayer{metric}', 'PlayerGameNumber']], 
                  on=['Player', 'GameId'], how='left')
    
    return df

def add_team_game_rolling_avg(df: pd.DataFrame, metric: str):
    """
    Calculates the rolling EWM average for each team in each game.
    Must have columns:
    Team, GameId, CumulativeTeam{metric}
    """
    df = df.sort_values('Date')
    home_stats = df.groupby(['HomeTeam', 'GameId', 'Date'])[[f'CumulativeTeam{metric}', 
                                                             f'CumulativeOpponent{metric}']].max().reset_index()
    home_stats[f'RollingAvgHomeTeam{metric}'] = home_stats.sort_values('Date') \
                                                             .groupby(['HomeTeam'])[[f'CumulativeTeam{metric}']] \
                                                             .transform(lambda x: x.ewm(span=5).mean())
    home_stats[f'RollingAvgHomeTeamAllowed{metric}'] = home_stats.sort_values('Date') \
                                                                    .groupby(['HomeTeam'])[[f'CumulativeOpponent{metric}']] \
                                                                    .transform(lambda x: x.ewm(span=5).mean())

    away_stats = df.groupby(['AwayTeam', 'GameId', 'Date'])[[f'CumulativeTeam{metric}', 
                                                             f'CumulativeOpponent{metric}']].max().reset_index()
    away_stats[f'RollingAvgAwayTeam{metric}'] = away_stats.sort_values('Date') \
                                                             .groupby(['AwayTeam'])[[f'CumulativeTeam{metric}']] \
                                                             .transform(lambda x: x.ewm(span=5).mean())
    away_stats[f'RollingAvgAwayTeamAllowed{metric}'] = away_stats.sort_values('Date') \
                                                                    .groupby(['AwayTeam'])[[f'CumulativeOpponent{metric}']] \
                                                                    .transform(lambda x: x.ewm(span=5).mean())

    df = df.merge(home_stats[['HomeTeam', 'GameId', f'RollingAvgHomeTeam{metric}', f'RollingAvgHomeTeamAllowed{metric}']],
                     left_on=['HomeTeam', 'GameId'], right_on=['HomeTeam', 'GameId'], how='left')
    df = df.merge(away_stats[['AwayTeam', 'GameId', f'RollingAvgAwayTeam{metric}', f'RollingAvgAwayTeamAllowed{metric}']], 
                    left_on=['AwayTeam', 'GameId'], right_on=['AwayTeam', 'GameId'], how='left')
    df[f'RollingAverageTeam{metric}'] = np.where(df['Home'] == 1, df[f'RollingAvgHomeTeam{metric}'], 
                                                                     df[f'RollingAvgAwayTeam{metric}'])
    df[f'RollingAverageOpposingTeamAllowed{metric}'] = np.where(df['Home'] == 0, 
                                                                   df[f'RollingAvgHomeTeamAllowed{metric}'], 
                                                                   df[f'RollingAvgAwayTeamAllowed{metric}'])
    
    return df

def stack_intervals(df: pd.DataFrame, intervals: int, metric: str):
    df_list = []
    for interval in range(intervals):
        interval_cols = [c for c in df.columns if f"{metric}_" in c]
        # only include columns from this interval and the other feature columns
        interval_cols_other = [c for c in interval_cols if f"{metric}_{interval}" not in c]
        this_interval_cols = [c for c in interval_cols if f"{metric}_{interval}" in c]
        temp_df = df[[c for c in df.columns if c not in interval_cols_other]]
        temp_df = temp_df.rename(columns={c: c.replace(f'{metric}_{interval}', f'{metric}Interval') for c in this_interval_cols})
        temp_df["Interval"] = interval + 1
        time_remaining = 48*60 - ((48/intervals) * (interval+1) * 60)
        time_remaining = 0 if (interval == (intervals-1)) else time_remaining
        time_remaining = time_remaining/60
        temp_df['ScoreMarginInterval'] = temp_df['CumulativeTeamPointsInterval'] - temp_df['CumulativeOpponentPointsInterval']
        temp_df['ScoreMarginxTimeRemainingInterval'] = abs(temp_df['ScoreMarginInterval']) * time_remaining
        temp_df['ScoreMarginxTimeRemaining2Interval'] = abs(temp_df['ScoreMarginInterval']) * time_remaining**2
        df_list.append(temp_df)
    # create synthetic for start of game
    start_df = temp_df.copy()
    start_df[this_interval_cols] = 0
    start_df["Interval"] = 0
    start_df['ScoreMarginInterval'] = 0
    start_df['ScoreMarginxTimeRemainingInterval'] = 0
    start_df['ScoreMarginxTimeRemaining2Interval'] = 0
    start_df[[c for c in start_df.columns if 'Cumulative' in c]] = 0
    df = pd.concat(df_list + [start_df], axis=0)

    return df


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--input_path", type=str, nargs='?', 
                        default="NBA_PBP_Datasets/NBA_PBP_2018-19.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, nargs='?', 
                        default="processed_data/nba_games_2018-19.csv", 
                        help="Path to save the processed CSV file")
    parser.add_argument("--intervals", type=int, nargs='?', default=4, help="How many intervals to split the game into")
    parser.add_argument("--remove_playoffs", type=bool, nargs='?', default=True, help="Whether to remove playoff games")
    parser.add_argument("--min_player_games", type=int, nargs='?', 
                        default=10, help="Minimum number of games a player must have played to be included")
    parser.add_argument("--min_player_avg_points", type=int, nargs='?', 
                        default=15, help="Minimum average points a player must have to be included")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create a dataframe with game features
    df = create_games_df(args.input_path, args.remove_playoffs)

    # Add player game scoring features
    df = add_player_game_scoring(df, args.intervals)

    # Add player game rolling average feature
    df = add_player_game_rolling_avg(df, metric='Points')

    # Add team game rolling average feature
    df = add_team_game_rolling_avg(df, metric='Points')

    # Stack the intervals
    df = df.groupby(['Player', 'GameId']).first().reset_index()
    df = stack_intervals(df, args.intervals, metric='Points')

    # Remove players first min_player_games, subset columns
    df = df.loc[df["PlayerGameNumber"] >= 10, PREPROCESSING_FEATURES]
    df = df.sort_values(by=['Player', 'GameId', 'Interval'], ascending=True)
    df = df.loc[df["RollingAvgPlayerPoints"] >= 15, PREPROCESSING_FEATURES]

    # Write the output dataframe to a CSV file
    df.to_csv(args.output_path, index=False)
