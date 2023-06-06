import argparse
from collections import Counter
import pandas as pd

def calculate_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        for index, row in df.iterrows():
            if row['exercise']:
                value += row['reward']
                break
    value = value/ngames
    return value

def calculate_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        reward = 0
        q_val = 0
        for index, row in df.iterrows():
            q_val = df.at[i*5, 'q_score']
            if row['exercise']:
                reward = row['reward']
                break
        value += reward - q_val
    value = value/ngames
    return value

def exercise_distr(data, ngames):
    values = []
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        for index, row in df.iterrows():
            if row['exercise']:
                values.append(index - (i*5)) 
                break
    values = dict(Counter(values))
    return values

def export_to_csv(data, file_name):
    file_exists = os.path.isfile(file_name)
    with open(file_name, "a", newline="") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data:
            writer.writerow(row)
            
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--scored_data_path", type=str, nargs='?', 
                        default="scored_data/scored_nba_games_2019-20.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--qvalfunction", type=str, nargs='?', 
                        default="DQN", 
                        help="Q Value Function")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the data
    scored_data = pd.read_csv(args.scored_data_path)
    nrows = len(scored_data)
    ngames = int(nrows/5)

    # Calculate the reward, profit, and exercise distribution
    results = [
        {"Reward": calculate_reward(scored_data, ngames), 
        "Profit": calculate_profit(scored_data, ngames), 
        "Distribution": exercise_distr(scored_data, ngames)}
    ]
    reward = calculate_reward(scored_data, ngames)
    profit = calculate_profit(scored_data, ngames)
    distr = exercise_distr(scored_data, ngames)

    # Normalize the features
    print(reward)
    print(profit)
    print(distr)
    #revenue of each strtegy and profit of each strategy, and what percent of time model has you excercise in a certain quarter
