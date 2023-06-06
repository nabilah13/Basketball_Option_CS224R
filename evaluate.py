import argparse
from collections import Counter
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

def qlearning_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        for index, row in df.iterrows():
            if row['exercise']:
                value += row['reward']
                break
    value = value/ngames
    return value

def qlearning_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        q_val = df.at[i*5, 'q_score']
        for index, row in df.iterrows():
            if row['exercise']:
                reward = row['reward']
                break
        value += reward - q_val
    value = value/ngames
    return value

def max_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        max = 0
        for index, row in df.iterrows():
            if row['reward'] > max:
                max = row['reward']
        value += max
    value = value/ngames
    return value

def max_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        max = 0
        q_val = df.at[i*5, 'q_score']
        for index, row in df.iterrows():
            if row['reward'] > max:
                max = row['reward']
        value += max - q_val
    value = value/ngames
    return value

def pos_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        reward = 0
        for index, row in df.iterrows():
            if row['reward'] > reward:
                reward = row['reward']
                break
        value += reward
    value = value/ngames
    return value

def pos_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        reward = 0
        q_val = df.at[i*5, 'q_score']
        for index, row in df.iterrows():
            if row['reward'] > reward:
                reward = row['reward']
                break
        value += reward - q_val
    value = value/ngames
    return value

def q2_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        value += df.at[(i*5)+2, 'reward']
    value = value/ngames
    return value

def q2_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        reward = df.at[(i*5)+2, 'reward']
        q_val = df.at[i*5, 'q_score']
        value += reward - q_val
    value = value/ngames
    return value

def q4_reward(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        value += df.at[(i*5)+4, 'reward']
    value = value/ngames
    return value

def q4_profit(data, ngames):
    value = 0
    for i in range(ngames):
        df = data[(i*5):((i*5)+5)]
        reward = df.at[(i*5)+4, 'reward']
        q_val = df.at[i*5, 'q_score']
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
                        default="scored_data/dqn/scored_nba_games_2019-20_corrected.csv", 
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

    # Calculate results
    results = [
        {"Q Value Function": args.qvalfunction,
        "Policy Reward": qlearning_reward(scored_data, ngames), 
        "Policy Profit": qlearning_profit(scored_data, ngames), 
        "Max Reward": max_reward(scored_data, ngames),
        "Max Profit": max_profit(scored_data, ngames),
        "Positive Reward": pos_reward(scored_data, ngames),
        "Positive Profit": pos_profit(scored_data, ngames),
        "Q2 Reward": q2_reward(scored_data, ngames),
        "Q2 Profit": q2_profit(scored_data, ngames),
        "Q4 Reward": q4_reward(scored_data, ngames),
        "Q4 Profit": q4_profit(scored_data, ngames)}
    ]

    # Output results to csv
    export_to_csv(results, "results.csv")

    # Create Distribution Plot
    distr = exercise_distr(scored_data, ngames)
    sorted_keys = sorted(distr.keys())
    sorted_values = [distr[key] for key in sorted_keys]
    total = sum(sorted_values)
    percentages = [value / total * 100 for value in sorted_values]
    bars = plt.bar(sorted_keys, percentages, color='lightblue')
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                f'{percentage:.1f}%', ha='center', va='bottom')
    plt.xticks(sorted_keys)
    plt.xlabel('Quarters')
    plt.ylabel('Percentage')
    plt.title('Distribution of {} Exercise Times'.format(args.qvalfunction))
    plt.savefig('charts/{}_distr_barchart.png'.format(args.qvalfunction))

    #print(results)
    #revenue of each strtegy and profit of each strategy, and what percent of time model has you excercise in a certain quarter
