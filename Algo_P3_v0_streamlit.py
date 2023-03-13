import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Chargement du dataset contenant les features des joueurs
data = pd.read_csv("https://github.com/nicolasmoreau10/Algo_P3/blob/main/DFBASE_V1.csv?raw=true", encoding='utf8', delimiter=',', low_memory=False)

lencoder = LabelEncoder()

data['surface_enc'] = lencoder.fit_transform(data['surface'])
data['tourney_level_enc'] = lencoder.fit_transform(data['tourney_level'])


X = data[['player_1_surface_winrate', 'player_2_surface_winrate', 'player_1_h2h',
'player_2_h2h', 'surface_enc', 'tourney_level_enc', 'player_1_bpFaced', 'player_2_bpFaced', 'player_1_svpt', 'player_2_svpt']]
# bpFaced : number of break points faced
# svpt : number of serve points

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# création du modèle
model = LogisticRegression(C = 0.1, max_iter = 100, penalty= 'l1', solver= 'liblinear').fit(X_train,y_train)


# STREAMLIT
st.title("Duel tennistique !")

# Sélection des joueurs
player1 = st.text_input("Joueur 1:")
player2 = st.text_input("Joueur 2:")

# Prédiction du gagnant
if st.button("Comparer les joueurs"):
    # Récupération des features des joueurs
    player1_features = data[data['player_1'] == player1][['player_1_surface_winrate', 'player_2_surface_winrate', 'player_1_h2h','player_2_h2h', 'surface_enc', 'tourney_level_enc', 'player_1_bpFaced', 'player_2_bpFaced', 'player_1_svpt', 'player_2_svpt']]
    player2_features = data[data['player_2'] == player2][['player_1_surface_winrate', 'player_2_surface_winrate', 'player_1_h2h','player_2_h2h', 'surface_enc', 'tourney_level_enc', 'player_1_bpFaced', 'player_2_bpFaced', 'player_1_svpt', 'player_2_svpt']]

    # Prédiction du gagnant
    winner = model.predict(np.array([player1_features.iloc[0], player2_features.iloc[0]]))
    
#     p1_stats = player1_features.values
#     p2_stats = player2_features.values
#     p1_prob = lr.predict_proba(p1_stats)[0][1]
#     p2_prob = lr.predict_proba(p2_stats)[0][1]

    # Affichage du résultat
    if winner[0] == player1:
        st.success(f"Le gagnant est {player1}")
    else :
        st.success(f"Le gagnant est {player2}")       
