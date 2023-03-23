import streamlit as st
import pandas as pd
import numpy as np


st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/photos-gratuite/court-tennis_93675-129141.jpg?w=1380&t=st=1679562557~exp=1679563157~hmac=20da8d2a005e02affb8ec4fa3f9d7f2015630db5dc0b5c82381fc6c3dda060a0");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


# STREAMLIT
st.markdown("<h1 style='text-align: center;'> DUEL TENNISTIQUE ! </h1>", unsafe_allow_html=True)

df=pd.read_csv("C:\\Users\\Administrateur\\Projet_3\\Algo_P3\\DFBASE_V1.csv")

def ace_analyze(nom_player): # La liste contient dans l'ordre: nombre moyen global d'ace, moyenne par Clay, Grass, Hard , Carpet
    surfaces = ['Clay', 'Grass', 'Hard', 'Carpet']
    liste=[]
    dw=df[(df['player_1'] == nom_player)]
    ace_winning= round(dw['player_1_ace'].mean(),1)
    #st.write(f"Nombre moyen d'ace quand {nom_player1} gagne: {ace_winning}")
    
    dl= df[df['player_2'] == nom_player]
    ace_losing= round(dl['player_2_ace'].mean(),1)
    #st.write(f"Nombre moyen d'ace quand {nom_player1} perd: {ace_losing}\n")
    moyenne= (ace_winning+ace_losing)/2
    liste.append(moyenne)
    for surface in surfaces:
        dw=df[(df['player_1'] == nom_player)]
        dw=dw[dw['surface'] == surface]
        ace_winning= round(dw['player_1_ace'].mean(),1)
        #st.write(f"Nombre moyen d'ace sur {surface} quand {nom_player1} gagne: {ace_winning}")

        dl= df[df['player_2'] == nom_player]
        dl=dl[dl['surface'] == surface]
        ace_losing= round(dl['player_2_ace'].mean(),1)
        #st.write(f"Nombre moyen d'ace sur {surface} quand {nom_player1} perd: {ace_losing}")
        moyenne_surf= (ace_winning+ace_losing)/2
        liste.append(moyenne_surf)
    return liste

def tx_vic_surface(nom_player, surface):
    #surfaces : 'Clay', 'Grass', 'Hard', 'Carpet'

    dj=df[((df['player_1'] == nom_player) | (df['player_2'] == nom_player)) & (df['winner'] == nom_player)]  
    dj= dj[dj['surface'] == surface]
    vic_surface= len(dj)  #Nombre de win sur la surface

    dj=df[(df['player_1'] == nom_player) | (df['player_2'] == nom_player)]
    dj= dj[dj['surface']== surface]
    total_surface= len(dj)  #Nombre total de matchs

    try:
        taux_victoire_surface =  round(vic_surface/(total_surface)*100,2)   # Cela renvoie un pourcentage de Victoire sur la surface 
    except:
        taux_victoire_surface=0
    return taux_victoire_surface


def best_surface_player(nom_player):
    surfaces = ['Clay', 'Grass', 'Hard', 'Carpet']
    top_surface= "indéfini"
    top_winrate=0

    for surface in surfaces:
        taux_victoire= tx_vic_surface(nom_player, surface)
        if taux_victoire > top_winrate:
            top_winrate= taux_victoire
            top_surface=surface

    st.write(f"La meilleure surface pour {nom_player} est {top_surface} avec un taux de Victoire de {top_winrate}%")
    

def player_hand(nom_player):
    df_player = df.loc[df['player_1'] == nom_player]
    main_player= df_player['player_1_hand'].values[0]
    if main_player =="L":
        return "Gaucher"
    elif main_player =="R":
        return "Droitier"
    elif main_player =="U":
        return "Latéralité inconnue"

def dernier_classement(nom_player):    #CETTE FONCTION RENVOIE LE DERNIER CLASSEMENT CONNU. NE PAS L'UTILISER POUR AVOIR LE CLASSEMENT ACTUEL CAR CERTAINS player1S SONT A LA RETRAITES
    df_player = df[(df['player_1'] == nom_player) | (df['player_2'] == nom_player)]
    df_player = df_player.sort_values('tourney_date', ascending=False)
    classement= df_player.iloc[0]
    if classement['player_1'] == nom_player:
        return df_player.iloc[0]['player_1_rank']
    else:
        return df_player.iloc[0]['player_2_rank']



def taille_age_natio(nom_player):
    df_player = df.loc[df['player_1'] == nom_player]
    taille= df_player['player_1_ht'].values[0]
    age= df_player['player_1_age'].max()
    natio= df_player['player_1_ioc'].values[0]
    
    return [taille,age,natio]

def nb_win_lose(nom_player):
    df_win =df[((df['player_1'] == nom_player) | (df['player_2'] == nom_player)) & (df['winner'] == nom_player)]  
    df_win=len(df_win)

    df_lose =df[((df['player_1'] == nom_player) | (df['player_2'] == nom_player)) & (df['winner'] != nom_player)]
    df_lose=len(df_lose)

    return [df_win,df_lose]


##########################################################  SELECTION JOUEUR #########################################################  ######################################################################################################################################

df_unique = df.groupby('player_1').first()
df_unique = df.groupby('player_1')['player_1'].first()
liste_player = df_unique.values.tolist()
liste_player.insert(0, 'Sélectionnez un joueur')

col1, col2, col3 = st.columns(3)

player1 = col1.selectbox("Choisissez le premier joueur", options= liste_player, key= 'A')
player2 = col2.selectbox("Choisissez le second joueur", options= liste_player, key="B")
choix_surfaces = col3.selectbox("Choisissez une surface", options = ['Choisissez une surface', 'Toutes', 'Clay', 'Grass', 'Hard'], key="C")

if st.button('Réinitialiser', key='D'):
    player1 = 'Sélectionnez un joueur'
    player2 = 'Sélectionnez un joueur'
    choix_surfaces = 'Choisissez une surface'
    
if not ((player1 == "Sélectionnez un joueur") or (player2 == "Sélectionnez un joueur") or (choix_surfaces == "Choisissez une surface")) :


    col1, col2 = st.columns(2)

    col1.title(f"{player1}")
    col1.image('https://www.tennisabstract.com/photos/' + player1.lower().replace(" ", "_") + '-sirobi' + '.jpg')
    col1.write(player_hand(player1))
    info=taille_age_natio(player1)
    col1.write(f"Taille: {info[0]} cm ")
    col1.write(f"Âge: {info[1]} ans")
    col1.write(f"Nationalité: {info[2]}")


    col2.title(f"{player2}")
    col2.image('https://www.tennisabstract.com/photos/' + player2.lower().replace(" ", "_") + '-sirobi' + '.jpg')
    col2.write(player_hand(player2))
    info=taille_age_natio(player2)
    col2.write(f"Taille: {info[0]} cm ")
    col2.write(f"Âge: {info[1]} ans")
    col2.write(f"Nationalité: {info[2]}")



    ##########################################################  COMPARAISON ##############################################################  ######################################################################################################################################

    st.markdown("<h1 style='text-align: center;'> VICTOIRES / DEFAITES </h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    result_j1= nb_win_lose(player1)
    result_j2= nb_win_lose(player2)

    col1.metric(label="VICTOIRES", value= result_j1[0])
    col1.metric(label="DEFAITES", value= result_j1[1])

    col2.metric(label="VICTOIRES", value= result_j2[0])
    col2.metric(label="DEFAITES", value= result_j2[1])



    st.markdown("<h1 style='text-align: center;'> ACES/MATCH </h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    ace1= ace_analyze(player1)
    ace2= ace_analyze(player2)

    deltaAce1 = ace1[0]-ace2[0]
    col1.metric(label="Moyenne d'ace/Match", value= ace1[0], delta=deltaAce1)

    deltaAce2 = ace2[0]-ace1[0]
    col2.metric(label="Moyenne d'ace/Match", value= ace2[0], delta=deltaAce2)



    st.markdown("<h1 style='text-align: center;'> WINRATE PAR SURFACE </h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    surfaces = ['Clay', 'Grass', 'Hard', 'Carpet']

    taux_victoire_player1 = []#sert pour le graph après
    taux_victoire_player2 = [] 

    for surface in surfaces:
        txj1= tx_vic_surface(player1, surface)
        txj2= tx_vic_surface(player2, surface)

        deltaTx = round(txj1-txj2,1)
        col1.metric(label= f"Winrate sur {surface}", value= f"{txj1}%", delta=f"{deltaTx}%")
        deltaTx2= round(txj2-txj1,1)
        col2.metric(label=f"Winrate sur {surface}", value= f"{txj2}%", delta=f"{deltaTx2}%")

        taux_victoire_player1.append(txj1)
        taux_victoire_player2.append(txj2) 




    ##########################################################  RADARS ###################################################################  ######################################################################################################################################


    import plotly.graph_objects as go
    surfaces = ['Clay', 'Grass', 'Hard']


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=taux_victoire_player1,
        theta=surfaces,
        fill='toself',
        name=f'Taux de victoire {player1}',))


    fig.add_trace(go.Scatterpolar(r=taux_victoire_player2,
        theta=surfaces,
        fill='toself',
        name=F'Taux de victoire {player2}', line=dict(color='rgba(255, 0, 0, 0.5)')))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 100]
        )),
      showlegend=True
    )

    st.plotly_chart(fig)    


    #####################################################  MACHINE LEARNING ##############################################################  ######################################################################################################################################


    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    # Chargement du dataset contenant les features des players

    lencoder = LabelEncoder()

    df['surface_enc'] = lencoder.fit_transform(df['surface'])
    df['tourney_level_enc'] = lencoder.fit_transform(df['tourney_level'])


    if choix_surfaces == 'Toutes':
        X = df[['player_1_surface_winrate', 'player_1_svpt', 'player_1_bpFaced', 'player_2_surface_winrate',  'player_2_svpt',  'player_2_bpFaced', 'surface_enc', 'tourney_level_enc']]
        y = df['target']
    else :    
        X = df[df['surface'] == choix_surfaces][['player_1_surface_winrate', 'player_1_svpt', 'player_1_bpFaced', 'player_2_surface_winrate',  'player_2_svpt',  'player_2_bpFaced', 'surface_enc', 'tourney_level_enc']]
        y = df[df['surface'] == choix_surfaces]['target']
    # bpFaced : number of break points faced
    # svpt : number of serve points

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle= True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # création du modèle
    model = LogisticRegression(C = 0.1, max_iter = 100, penalty= 'l1', solver= 'liblinear').fit(X_train,y_train)



    # charger le dataframe de moyennes

    moy_df = pd.read_csv("C:\\Users\\Administrateur\\Projet_3\\Algo_P3\\DFMOYV5b.csv")

    # Prédiction du gagnant


    st.markdown("<h1 style='text-align: center;'> AND NOW, LADIES AND GENTLEMEN... </h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    bouton = col2.button("MATCH !")

    if bouton :

            player1_features = moy_df[moy_df['Joueur'] == player1][[choix_surfaces,'svpt','bpFaced']]
            player2_features = moy_df[moy_df['Joueur'] == player2][[choix_surfaces,'svpt','bpFaced']]
            player1_array = player1_features.iloc[0].to_numpy()
            player2_array = player2_features.iloc[0].to_numpy()


            concatenated_array = np.concatenate([player1_array, player2_array, [4, 2]])
            winner = model.predict(concatenated_array.reshape(1, -1))

            # afficher le résultat
            if winner == player1:
                st.success(f"Le gagnant est {player1}")
            else:
                st.success(f"Le gagnant est {player2}")