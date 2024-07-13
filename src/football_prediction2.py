import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def main():
    print("Starter programmet...")

    # Sett opp filbaner
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, 'datasets', 'results.csv')
    games_to_be_played_path = os.path.join(base_path, 'datasets', 'games_to_be_played.csv')
    fifa_rankings_path = os.path.join(base_path, 'datasets', 'fifa_ranking.csv')

    # Last inn data
    print("Laster inn data...")
    results = pd.read_csv(results_path)
    games_to_be_played = pd.read_csv(games_to_be_played_path)
    fifa_rankings = pd.read_csv(fifa_rankings_path)

    # Filtrer data etter år 1970
    print("Filtrerer data etter år 1970...")
    results['date'] = pd.to_datetime(results['date'])
    results = results[results['date'] >= '1970-01-01']

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

    # Gjør prediksjoner på testsettet
    print("Gjennomfører prediksjoner på testsettet...")
    y_pred_home = model_home.predict(X_test)
    y_pred_away = model_away.predict(X_test)

    # Evaluer modellen
    print("Evaluerer modellen...")
    mse_home = mean_squared_error(y_test_home, y_pred_home)
    mse_away = mean_squared_error(y_test_away, y_pred_away)
    print(f'MSE Home: {mse_home}, MSE Away: {mse_away}')

    # Prediksjon av nye kamper
    print("Forbereder prediksjon av nye kamper...")
    games_to_be_played['home_team_rank'] = games_to_be_played.apply(lambda row: get_team_rank_on_date(row['home_team'], pd.to_datetime('today')), axis=1)
    games_to_be_played['away_team_rank'] = games_to_be_played.apply(lambda row: get_team_rank_on_date(row['away_team'], pd.to_datetime('today')), axis=1)
    games_to_be_played = games_to_be_played.dropna(subset=['home_team_rank', 'away_team_rank'])

    # Lag features for prediksjon
    features_to_predict = games_to_be_played[['home_team_rank', 'away_team_rank']]
    features_to_predict.loc[:, 'days_since'] = 0  # Siden prediksjonene er for dagens dato
    features_to_predict['home_advantage'] = 1  # Anta hjemmefordel for alle nye kamper

    # Gjør prediksjoner
    print("Gjennomfører prediksjoner for nye kamper...")
    predicted_home_scores = model_home.predict(features_to_predict)
    predicted_away_scores = model_away.predict(features_to_predict)

    # Rund av til nærmeste heltall
    predicted_home_scores = np.round(predicted_home_scores).astype(int)
    predicted_away_scores = np.round(predicted_away_scores).astype(int)

    # Legg prediksjonene tilbake til DataFrame
    games_to_be_played['predicted_home_score'] = predicted_home_scores
    games_to_be_played['predicted_away_score'] = predicted_away_scores

    print("Genererer PDF med prediksjonene...")
    generate_pdf(games_to_be_played[['home_team', 'away_team', 'predicted_home_score', 'predicted_away_score']])
    print("PDF generert!")

def generate_pdf(predictions):
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.drawString(30, height - 30, "Predikerte resultater for kommende kamper")
    c.drawString(30, height - 50, "-"*80)

    y = height - 70
    for index, row in predictions.iterrows():
        line = f"{row['home_team']} vs {row['away_team']} - {row['predicted_home_score']} : {row['predicted_away_score']}"
        c.drawString(30, y, line)
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 30

    c.save()

if __name__ == "__main__":
    main()
