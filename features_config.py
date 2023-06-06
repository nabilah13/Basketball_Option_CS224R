PREPROCESSING_FEATURES = [
    "Interval",
    "CumulativePlayerPointsInterval",
    "Player",
    "GameId",
    "Date",
    "Team",
    "Opponent",
    "Home",
    "TeamRest",
    "OpponentTeamRest",
    "CumulativeTeamPointsInterval",
    "CumulativeOpponentPointsInterval",
    "ScoreMarginInterval",
    "ScoreMarginxTimeRemainingInterval",
    "ScoreMarginxTimeRemaining2Interval",
    "RollingAvgPlayerPoints",
    "RollingAverageOpposingTeamAllowedPoints",
    "RollingAverageTeamPoints",
]

TRAINING_FEATURES = [
    "Interval",
    "CumulativePlayerPointsInterval",
    "Home",
    "TeamRest",
    "OpponentTeamRest",
    "CumulativeTeamPointsInterval",
    "CumulativeOpponentPointsInterval",
    "ScoreMarginInterval",
    "ScoreMarginxTimeRemainingInterval",
    "ScoreMarginxTimeRemaining2Interval",
    "RollingAvgPlayerPoints",
    "RollingAverageOpposingTeamAllowedPoints",
    "RollingAverageTeamPoints",
]

FEATURES_TO_NORMALIZE = [
    "ScoreMarginInterval",
    "ScoreMarginxTimeRemainingInterval",
    "ScoreMarginxTimeRemaining2Interval",
    "RollingAverageOpposingTeamAllowedPoints",
    "RollingAverageTeamPoints",
]