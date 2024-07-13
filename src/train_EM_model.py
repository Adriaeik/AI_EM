import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

def main():
    print("Starter programmet...")

    # Sett opp filbaner
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, 'datasets', 'results.csv')
    fifa_rankings_path = os.path.join(base_path, 'datasets', 'fifa_ranking.csv')

    # Last inn data
    print("Laster inn data...")
    results = pd.read_csv(results_path)
    fifa_rankings = pd.read_csv(fifa_rankings_path)

    # Filtrer data etter år 1970
    print("Filtrerer data etter år 1900...")
    results['date'] = pd.to_datetime(results['date'])
    results = results[results['date'] >= '1900-01-01']

    # Preprocess FIFA rankings data
    print("Forbereder FIFA-rangeringer...")
    fifa_rankings['rank_date'] = pd.to_datetime(fifa_rankings['rank_date'])

    # Funksjon for å slå sammen FIFA-rangeringer med kampresultater
    def get_team_rank_on_date(team, date):
        relevant_rankings = fifa_rankings[(fifa_rankings['country_full'] == team) & (fifa_rankings['rank_date'] <= date)]
        if not relevant_rankings.empty:
            return relevant_rankings.sort_values(by='rank_date', ascending=False).iloc[0]['rank']
        else:
            return None

    print("Kombinerer data med FIFA-rangeringer...")
    results['home_team_rank'] = results.apply(lambda row: get_team_rank_on_date(row['home_team'], row['date']), axis=1)
    results['away_team_rank'] = results.apply(lambda row: get_team_rank_on_date(row['away_team'], row['date']), axis=1)

    # Filtrer bort kamper uten tilgjengelig FIFA-rangering
    results = results.dropna(subset=['home_team_rank', 'away_team_rank'])

    # Lag features og labels
    print("Lager features og labels...")
    features = results[['home_team_rank', 'away_team_rank', 'date']]
    features.loc[:, 'days_since'] = (pd.to_datetime('today') - features['date']).dt.days
    features.loc[:, 'home_advantage'] = results['neutral'].apply(lambda x: 0 if x else 1)
    features = features.drop(columns=['date'])
    labels_home = results['home_score']
    labels_away = results['away_score']

    # Del data i trenings- og testsett
    print("Deler data i trenings- og testsett...")
    X_train, X_test, y_train_home, y_test_home = train_test_split(features, labels_home, test_size=0.2, random_state=42)
    _, _, y_train_away, y_test_away = train_test_split(features, labels_away, test_size=0.2, random_state=42)

    # Tren en MLPRegressor modell
    print("Trener MLPRegressor modell for hjemmepoeng...")
    model_home = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
    model_home.fit(X_train, y_train_home)

    print("Trener MLPRegressor modell for bortepoeng...")
    model_away = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
    model_away.fit(X_train, y_train_away)

    # Lagre modellene
    print("Lagrer modellene...")
    joblib.dump(model_home, os.path.join(base_path, 'model_home.pkl'))
    joblib.dump(model_away, os.path.join(base_path, 'model_away.pkl'))

    print("Modellene er trent og lagret.")

if __name__ == "__main__":
    main()
