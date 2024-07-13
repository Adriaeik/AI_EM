import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Sett opp filbaner
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_path, 'datasets', 'results.csv')
games_to_be_played_path = os.path.join(base_path, 'datasets', 'games_to_be_played.csv')
fifa_rankings_path = os.path.join(base_path, 'datasets', 'fifa_ranking.csv')

# Last inn data
results = pd.read_csv(results_path)
games_to_be_played = pd.read_csv(games_to_be_played_path)
fifa_rankings = pd.read_csv(fifa_rankings_path)

# Konverter dato til datetime og filtrer data etter 2000
results['date'] = pd.to_datetime(results['date'])
results = results[results['date'] >= '2000-01-01']

# Forbered FIFA-rangeringer
fifa_rankings['rank_date'] = pd.to_datetime(fifa_rankings['rank_date'])
latest_rankings = fifa_rankings.loc[fifa_rankings.groupby('country_full')['rank_date'].idxmax()]
latest_rankings = latest_rankings[['country_full', 'rank']].set_index('country_full')
latest_rankings.index = latest_rankings.index.str.lower()  # For å matche med team names

# Funksjon for å lage teamstatistikk
def calculate_team_stats(results):
    results['goal_difference'] = results['home_score'] - results['away_score']
    home_stats = results.groupby('home_team').agg({
        'home_score': 'mean', 
        'away_score': 'mean', 
        'goal_difference': 'mean',
        'home_team': 'count'
    }).rename(columns={
        'home_score': 'home_goals_for', 
        'away_score': 'home_goals_against', 
        'goal_difference': 'home_goal_diff',
        'home_team': 'home_matches_played'
    })
    away_stats = results.groupby('away_team').agg({
        'home_score': 'mean', 
        'away_score': 'mean', 
        'goal_difference': 'mean',
        'away_team': 'count'
    }).rename(columns={
        'home_score': 'away_goals_against', 
        'away_score': 'away_goals_for', 
        'goal_difference': 'away_goal_diff',
        'away_team': 'away_matches_played'
    })
    return home_stats, away_stats

home_stats, away_stats = calculate_team_stats(results)

# Funksjon for å legge til tidsvekter
def add_time_weights(results):
    current_date = datetime.now()
    results['days_since'] = (current_date - results['date']).dt.days
    results['weight'] = np.exp(-results['days_since'] / 365.0)  # Eksponentiell vekting
    return results

results = add_time_weights(results)

# Forbered data for modellering
features = ['home_goals_for', 'home_goals_against', 'home_goal_diff', 'home_matches_played',
            'away_goals_for', 'away_goals_against', 'away_goal_diff', 'away_matches_played',
            'home_fifa_rank', 'away_fifa_rank']

# Kombiner statistikkene for hjemme- og bortelag
team_stats = home_stats.join(away_stats, how='outer').fillna(0)

results = results.join(team_stats, on='home_team')
results = results.join(team_stats, on='away_team', rsuffix='_away')

# Legg til FIFA-rangeringer
results['home_team_lower'] = results['home_team'].str.lower()
results['away_team_lower'] = results['away_team'].str.lower()
results = results.join(latest_rankings, on='home_team_lower')
results = results.join(latest_rankings, on='away_team_lower', rsuffix='_away')

results = results.rename(columns={'rank': 'home_fifa_rank', 'rank_away': 'away_fifa_rank'})

# Fyll manglende rangeringer med en høy verdi (f.eks. 200)
results['home_fifa_rank'] = results['home_fifa_rank'].fillna(200)
results['away_fifa_rank'] = results['away_fifa_rank'].fillna(200)

# Forbered input og output for modell
X = results[features]
y_home = results['home_score']
y_away = results['away_score']

# Split data i trenings- og testsett
X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(
    X, y_home, y_away, test_size=0.2, random_state=42
)

# Tren modellene
model_home = GradientBoostingRegressor()
model_away = GradientBoostingRegressor()

model_home.fit(X_train, y_train_home, sample_weight=results.loc[X_train.index, 'weight'])
model_away.fit(X_train, y_train_away, sample_weight=results.loc[X_train.index, 'weight'])

# Evaluer modellene
preds_home = model_home.predict(X_test)
preds_away = model_away.predict(X_test)

# Rund av prediksjonene til nærmeste heltall
preds_home_rounded = np.round(preds_home).astype(int)
preds_away_rounded = np.round(preds_away).astype(int)

# Beregn og skriv ut evaluering metrikker
print("Rounded Mean Squared Error for Home Score:", mean_squared_error(y_test_home, preds_home_rounded))
print("Rounded Mean Absolute Error for Home Score:", mean_absolute_error(y_test_home, preds_home_rounded))

print("Rounded Mean Squared Error for Away Score:", mean_squared_error(y_test_away, preds_away_rounded))
print("Rounded Mean Absolute Error for Away Score:", mean_absolute_error(y_test_away, preds_away_rounded))

# Forutsig resultater for kommende kamper
def predict_upcoming_games(games, team_stats, latest_rankings, model_home, model_away):
    results = []

    for index, row in games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        game = pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]})
        game = game.join(team_stats, on='home_team')
        game = game.join(team_stats, on='away_team', rsuffix='_away')

        # Legg til FIFA-rangeringer
        game['home_team_lower'] = game['home_team'].str.lower()
        game['away_team_lower'] = game['away_team'].str.lower()
        game = game.join(latest_rankings, on='home_team_lower')
        game = game.join(latest_rankings, on='away_team_lower', rsuffix='_away')

        game = game.rename(columns={'rank': 'home_fifa_rank', 'rank_away': 'away_fifa_rank'})

        # Fyll manglende rangeringer med en høy verdi (f.eks. 200)
        game['home_fifa_rank'] = game['home_fifa_rank'].fillna(200)
        game['away_fifa_rank'] = game['away_fifa_rank'].fillna(200)

        X_game = game[features]

        pred_home = model_home.predict(X_game)[0]
        pred_away = model_away.predict(X_game)[0]

        pred_home = np.round(pred_home).astype(int)
        pred_away = np.round(pred_away).astype(int)

        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_score': pred_home,
            'predicted_away_score': pred_away
        })

    return pd.DataFrame(results)

# Predict results for games to be played
predicted_games = predict_upcoming_games(games_to_be_played, team_stats, latest_rankings, model_home, model_away)
print(predicted_games)