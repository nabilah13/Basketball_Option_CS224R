import argparse
import pandas as pd
import torch

from dataset import create_basketball_dataloader
from models import DQN_model
from features_config import TRAINING_FEATURES, FEATURES_TO_NORMALIZE
from tqdm import tqdm


def train_dqn_model(num_epochs: int, train_dataloader, valid_dataloader, model: DQN_model, skip_validation: int = 0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        train_progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                            desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch")
        epoch_loss = 0.0
        num_batches = 0
        for _, batch in train_progress_bar:
            state, reward, next_state, terminal, _, _, _ = batch
            # compute q score
            q_score = model(state).flatten()
            q_score_next = model(next_state).flatten()
            # detach q_score_next, set to zero if terminal
            q_score_next = q_score_next.detach()
            q_score_next[terminal] = 0
            # compute loss
            loss = torch.mean((torch.max(reward, q_score_next) - q_score)**2)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()
            # update progress bar
            epoch_loss += loss.item()
            num_batches += 1
            train_progress_bar.set_postfix({"Batch Loss": loss.item()})
        # Calculate the average loss for the epoch TO DO FIX THIS
        average_epoch_loss = epoch_loss / num_batches
        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Train Average Loss: {average_epoch_loss:.4f}")

        if skip_validation==0:
            # Calculate the validation loss for the epoch
            valid_progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader),
                                        desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch")
            valid_loss = 0.0
            num_batches = 0
            for _, batch in valid_progress_bar:
                with torch.no_grad():
                    state, reward, next_state, terminal, _, _, _ = batch
                    # compute q score
                    q_score = model(state).flatten()
                    q_score_next = model(next_state).flatten()
                    # set to zero if terminal
                    q_score_next[terminal] = 0
                    # compute loss
                    loss = torch.mean((torch.max(reward, q_score_next) - q_score)**2)
                    # update progress bar
                    valid_loss += loss.item()
                    num_batches += 1
                    valid_progress_bar.set_postfix({"Batch Loss": loss.item()})
            # Calculate the average loss for the epoch
            average_valid_loss = valid_loss / num_batches
            # Print the average loss for the epoch
            print(f"Epoch {epoch+1}/{num_epochs} - Valid Average Loss: {average_valid_loss:.4f}")

    return model 

def score_validation_df(valid_dataloader, model: DQN_model):
    policy_dfs = list()
    valid_progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader),
                                desc=f"Scoring Validation Set", unit="batch")
    
    for _, batch in valid_progress_bar:
        with torch.no_grad():
            state, reward, next_state, terminal, gameId, player, interval = batch
            # compute q score
            q_score = model(state).flatten()
            # exercise true if terminal is true or if reward is greater than q_score_next
            exercise = terminal | (reward > q_score)
            # add player, gameId, and exercise to the policy_data
            policy_dfs.append(pd.DataFrame({"interval": interval,  "gameId": gameId, 
                                            "player": player, "exercise": exercise, 
                                            "q_score": q_score, "reward": reward}))
    
    policy_df = pd.concat(policy_dfs, ignore_index=True)

    return policy_df
            
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--input_train_path", type=str, nargs='?', 
                        default="processed_data/nba_games_2018-19.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--input_valid_path", type=str, nargs='?', 
                        default="processed_data/nba_games_2019-20.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--scored_valid_path", type=str, nargs='?', 
                        default="scored_data/scored_nba_games_2019-20.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--num_intervals", type=int, nargs='?',
                        default=4,
                        help="Number of intervals to use for the basketball option")
    parser.add_argument("--num_epochs", type=int, nargs='?',
                        default=5,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, nargs='?',
                        default=64,
                        help="Batch size for training")
    parser.add_argument("--lspi_or_dqn", type=int, nargs='?', default=1,
                        help="0 for LSPI, 1 for DQN")
    parser.add_argument("--skip_validation", type=int, nargs='?', default=0,
                        help="0 to score validation set, 1 to skip")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the data
    train_df = pd.read_csv(args.input_train_path)
    valid_df = pd.read_csv(args.input_valid_path)

    # Normalize the features
    features_means = train_df[FEATURES_TO_NORMALIZE].mean()
    feature_stds = train_df[FEATURES_TO_NORMALIZE].std()
    train_df[FEATURES_TO_NORMALIZE] = (train_df[FEATURES_TO_NORMALIZE] - features_means) / feature_stds
    valid_df[FEATURES_TO_NORMALIZE] = (valid_df[FEATURES_TO_NORMALIZE] - features_means) / feature_stds

    # Create the dataloaders
    train_dataloader = create_basketball_dataloader(train_df, batch_size=args.batch_size, 
                                                    shuffle=True, num_intervals=args.num_intervals)
    valid_dataloader = create_basketball_dataloader(valid_df, batch_size=args.batch_size, 
                                                    shuffle=False, num_intervals=args.num_intervals)

    # Create the model
    dqn_model = DQN_model(len(TRAINING_FEATURES))
    dqn_model = train_dqn_model(args.num_epochs, train_dataloader, valid_dataloader, dqn_model, args.skip_validation)

    # Score the validation set
    policy_df = score_validation_df(valid_dataloader, dqn_model)
    policy_df.to_csv(args.scored_valid_path, index=False)
