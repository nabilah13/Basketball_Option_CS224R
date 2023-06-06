import pandas as pd
import numpy as np
import torch

from features_config import TRAINING_FEATURES
from torch.utils.data import Dataset, DataLoader

class BasketballDataset(Dataset):
    def __init__(self, input_df: pd.DataFrame, intervals: int = 4):
        self.data = input_df.dropna()
        self.intervals = intervals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        current_row = self.data.iloc[index, :]
        next_row = self.data.iloc[min(index+1,len(self.data)-1), :]

        player = current_row["Player"]
        gameId = current_row["GameId"]

        current_interval = current_row["Interval"]
        strike = current_row["RollingAvgPlayerPoints"] * current_interval / self.intervals
        exercise_payout = current_row["CumulativePlayerPointsInterval"] - strike
        reward = torch.max(torch.tensor(exercise_payout), torch.tensor(0))

        if current_interval == self.intervals:
            terminal = True
        else:
            terminal = False

        state = torch.tensor(current_row[TRAINING_FEATURES].values.astype(np.float32))
        next_state = torch.tensor(next_row[TRAINING_FEATURES].values.astype(np.float32))

        return state, reward, next_state, terminal, gameId, player, current_interval
    
def create_basketball_dataloader(input_df: pd.DataFrame, batch_size: int = 32, shuffle: bool = True, 
                                 num_workers: int = 0, num_intervals: int = 4):
    dataset = BasketballDataset(input_df, num_intervals)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
    
if __name__ == "__main__":
    input_path = "processed_data/nba_games_2018-19.csv"
    input_df = pd.read_csv(input_path)
    train_dataloader = create_basketball_dataloader(input_df)

    # Print five rows of dataloader as test
    for i in range(5):
        print(next(iter(train_dataloader)))