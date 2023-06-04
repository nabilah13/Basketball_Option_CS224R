# Basketball_Option_CS224R

Reinforcement learning has emerged as an intriguing strategy for pricing financial derivatives, such as American options. By framing the pricing of American options as a type of optimal stopping time problem, we can apply Q-learning methods such a LSPI (least-squares policy iteration) and DQN (Deep Q Networks). 

To add an additional fun twist, we create and price a basketball option based on NBA games. Our basketball option allows bettors to pay a fixed price at the beginning of a game to bet on the performance of a particular player, with option exercise allowable at predefined points in time (similar to a Bermuda option). As an example, we allow option exercise during the quarter breaks. 

The exercise payout is the difference between the points scored by the player of interest and their expected points scored. The expected points scored is calculated as the player's average points over the last 10 games scaled linearly for how many time intervals have elapsed in the current game. For example, if Lebron James has averaged 30 points per game over the prior 10 games, then his expected points scored at halftime would be 15 points.

### Dataset
The starting NBA play-by-play dataset was scraped from Basketball Reference and is available publicly on Kaggle at the following link: https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019?select=NBA_PBP_2020-21.csv

The dataset is too large to host directly on GitHub, but it is hosted in full on Kaggle.

### Data Preprocessing
In data_preprocessing.py, we transform the play-by-play data from Basketball Reference into a dataset of NBA player-games with one row for each quarter of the game. We use "CumulativePlayerPointsInterval" and "RollingAvgPlayerPointsInterval" as the equivalent of the American option spot price and strike price, respectively. The difference between these two fields determines the exercise value of the basketball option.
