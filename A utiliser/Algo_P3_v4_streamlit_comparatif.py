import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64


path= "C:\\Users\\Administrateur\\Projet_3\\Algo_P3\\"

def background_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
background_local(path + "photos\\" + "fond-court-tennis.png")

col1, col2, col3 = st.columns([5.5,1.6,6])
balle = Image.open(path + "\\photos\\" + "balle.png")
col2.image(balle, width=100, use_column_width=True)


# STREAMLIT
st.markdown("<h1 style='text-align: center; color : #ac211e'> JEU, SET AND DASH ! </h1>", unsafe_allow_html=True)

st.write("")
st.write("")

df=pd.read_csv(path + "DFBASE_V1b.csv", sep =';')
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df['tourney_date'] = df['tourney_date'].dt.strftime('%d-%m-%Y')
# df['tourney_date'] = df['tourney_date'].apply(lambda x: x.strftime('%d-%m-%Y')) # Convertir en format de date française


def ace_analyze(nom_player): # La liste contient dans l'ordre: nombre moyen global d'ace, moyenne par Clay, Grass, Hard
    surfaces = ['Clay', 'Grass', 'Hard']
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
    #surfaces : 'Clay', 'Grass', 'Hard'

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
    surfaces = ['Clay', 'Grass', 'Hard']
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


def histo_duel(player1,player2):
    duels1= (df[(df['player_1_name'] == player1) & (df['player_2_name'] == player2)])  
    duels2= (df[(df['player_1_name'] == player2) & (df['player_2_name'] == player1)])
    duels= pd.concat([duels1,duels2])
    print("taille du dataframe duel:", len(duels))
    vicj1= len (duels[duels['winner']== player1])
    vicj2= len (duels[duels['winner']== player2])
    return vicj1,vicj2
##########################################################  SELECTION JOUEUR #########################################################  ######################################################################################################################################

df_unique = df.groupby('player_1').first()
df_unique = df.groupby('player_1')['player_1'].first()
liste_player = df_unique.values.tolist()
liste_player.insert(0, 'Sélectionnez un joueur')



col1, col2 = st.columns(2)

player1 = col1.selectbox("Choisissez le premier joueur", options= liste_player, key = 'A')
player2 = col2.selectbox("Choisissez le second joueur", options= liste_player, key = 'B')

# gestion cas même joueur pour duel

meme_joueur = False

if player1 == player2 and (player1 != "Sélectionnez un joueur" or player2 != "Sélectionnez un joueur") :
    meme_joueur = True
    st.markdown("<h2 style='text-align: center; color:red'>Vraiment ? ;)</h2>", unsafe_allow_html=True)
    

st.write("")
# if st.button('Réinitialiser', key='D'):
#     player1 = 'Sélectionnez un joueur'
#     player2 = 'Sélectionnez un joueur'
#     choix_surfaces = 'Choisissez une surface'
    
    
if not ((player1 == "Sélectionnez un joueur") or (player2 == "Sélectionnez un joueur")) and meme_joueur == False :
   
   
    col1, col2,col3 = st.columns([6,1,6])

    col1.markdown(f"<p style='color:black;text-align: center'>{player1}</p>", unsafe_allow_html=True)   
    try:
        imagej1 = Image.open(path + "\\photos\\" + player1.split(' ')[1].lower()+".png")
        col1.image(imagej1)
        print(path + player1.split(' ')[1].lower())
    except:
        imagej1 = Image.open(path + "\\photos\\" + "logo2.png")
        col1.image(imagej1)
        col1.write("")
    col1.write(player_hand(player1))
    info=taille_age_natio(player1)
    col1.write(f"Taille: {info[0]} cm ")
    col1.write(f"Âge: {info[1]} ans")
    col1.write(f"Nationalité: {info[2]}")

    col2.markdown("<h2 style='text-align: center; color : #ac211e'>VS</h2>", unsafe_allow_html=True)

    col3.markdown(f"<p style='color:black;text-align: center'>{player2}</p>", unsafe_allow_html=True)  
    try:
        imagej2= Image.open(path + "\\photos\\" + player2.split(' ')[1].lower() +".png")
        print(imagej2)
        col3.image(imagej2)
    except:
        imagej2 = Image.open(path + "\\photos\\" + "logo2.png")
        col3.image(imagej2)
        print(imagej2)
    col3.write(player_hand(player2))
    info=taille_age_natio(player2)
    col3.write(f"Taille: {info[0]} cm ")
    col3.write(f"Âge: {info[1]} ans")   
    col3.write(f"Nationalité: {info[2]}")

    historique= histo_duel(player1,player2)
    data= {player2:historique[-1],player1:historique[-2]}
    
        
    st.write("")
    
    if (historique[-1] == 0) and (historique[-2] == 0) :
        st.markdown("<h2 style='text-align: center; color:red'>Pas de passif entre ces 2 joueurs !</h2>", unsafe_allow_html=True)
    else :
        nb_duels = historique[-1] + historique[-2]
        duels1= (df[(df['player_1_name'] == player1) & (df['player_2_name'] == player2)])  
        duels2= (df[(df['player_1_name'] == player2) & (df['player_2_name'] == player1)])
        duels = pd.concat([duels1,duels2])
        duels['tourney_date'] = pd.to_datetime(duels['tourney_date'])
        duels['tourney_date'] = duels['tourney_date'].dt.year
        premier_duel = duels['tourney_date'].min()
        dernier_duel = duels['tourney_date'].max()
        st.markdown("<h1 style='text-align: center;'> HISTORIQUE TAUX VICTOIRE VS</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color : #ac211e'>Les joueurs se sont affrontés {nb_duels} fois entre {premier_duel} et {dernier_duel}.</h3>", unsafe_allow_html=True)
        fig = go.Figure(
        go.Pie(
        labels=list(data.keys()),
        values=list(data.values()),
        hole=0.5,
        marker={"colors": ["#008fd5", "#fc4f30"]}, ))
        fig.update_layout(
        font=dict(size=18),margin=dict(t=20, b=0),width=400, height=300, legend=dict(
        orientation="h",
        y=-0.1,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    

    ##########################################################  COMPARAISON ##############################################################  ######################################################################################################################################

    st.markdown("<h1 style='text-align: center;'> VICTOIRES / DEFAITES </h1>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6  = st.columns(6)

    result_j1= nb_win_lose(player1)
    result_j2= nb_win_lose(player2)

    col1, col2  = st.columns(2)
    
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=(result_j1[0]*100/(result_j1[0] + result_j1[1])),
        title={'text': f"{player1}"},
        number={'suffix': '%'},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue", 'thickness': 1},
               'steps': [{'range': [0, 100], 'color': "royalblue"}]}))
#     fig4.update_layout(
#         height=400,
#         width=600,
#     )
    col1.plotly_chart(fig4, use_container_width=True)
    
    fig5 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = (result_j2[0]*100/(result_j2[0] + result_j2[1])),
    title = {'text': f"{player2}"},
    number={'suffix': '%'},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "darkblue", 'thickness': 1},
             'steps' : [
                {'range': [0, 100], 'color': "royalblue"}]}))
#     fig5.update_layout(
#         height=400,
#         width=600,
#     )
    col2.plotly_chart(fig5, use_container_width=True)
 
    st.markdown("<h1 style='text-align: center;'> TAUX VICTOIRE PAR SURFACE </h1>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

    surfaces = ['Clay', 'Grass', 'Hard']

    taux_victoire_player1 = []#sert pour le graph après
    taux_victoire_player2 = [] 

    for surface in surfaces:
        txj1= tx_vic_surface(player1, surface)
        txj2= tx_vic_surface(player2, surface)

#         deltaTx = round(txj1-txj2,1)
#         col1.metric(label= f"Winrate sur {surface}", value= f"{txj1}%", delta=f"{deltaTx}%")
#         deltaTx2= round(txj2-txj1,1)
#         col2.metric(label=f"Winrate sur {surface}", value= f"{txj2}%", delta=f"{deltaTx2}%")

        taux_victoire_player1.append(txj1)
        taux_victoire_player2.append(txj2) 
        
    
        
        
    ##########################################################  RADARS ###################################################################  ######################################################################################################################################

    import plotly.express as px
    import plotly.graph_objects as go
    
    surfaces = ['Clay', 'Grass', 'Hard']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=taux_victoire_player1,theta=surfaces,fill='toself', name=f'Taux de victoire {player1}',))
    fig.add_trace(go.Scatterpolar(r=taux_victoire_player2,theta=surfaces,fill='toself',name=F'Taux de victoire {player2}', line=dict(color='rgba(255, 0, 0, 0.5)')))
    fig.update_layout(
    font=dict(size=15),margin=dict(t=20, b=0),width=500, height=350, legend=dict(
    orientation="h",
    y=-0.1,
    x=0.5,
    xanchor="center",
    yanchor="top"),
    polar=dict(radialaxis=dict(visible=True,range=[0, 100])
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    
    df_joueur = pd.read_csv(path + "DFSHOTSCHELEMMOD.csv", sep = ';')

    def doublepiechart(df_joueur,joueur):
        liste= []
        df=df_joueur
        df=df[df['joueur']==joueur]
        forehand_cross= df[df['row']=='F']['crosscourt'].mean()
        forehand_down_middle= df[df['row']=='F']['down_middle'].mean()
        forehand_down_the_line= df[df['row']=='F']['down_the_line'].mean()
        backhand_cross= df[df['row']=='B']['crosscourt'].mean()
        backhand_down_middle= df[df['row']=='B']['down_middle'].mean()
        backhand_down_the_line= df[df['row']=='B']['down_the_line'].mean()
        slice_cross= df[df['row']=='S']['crosscourt'].mean()
        slice_down_middle= df[df['row']=='S']['down_middle'].mean()
        slice_down_the_line= df[df['row']=='S']['down_the_line'].mean()
        liste.append(forehand_cross)
        liste.append(forehand_down_middle)
        liste.append(forehand_down_the_line)
        liste.append(backhand_cross)
        liste.append(backhand_down_middle)
        liste.append(backhand_down_the_line)
        liste.append(slice_cross)
        liste.append(slice_down_middle)
        liste.append(slice_down_the_line)
        return liste
    
    col1, col2 = st.columns(2)

    p1 = doublepiechart(df_joueur, player1)
    data = {'Shot Type': ['Forehand', 'Forehand', 'Forehand', 'Backhand', 'Backhand', 'Backhand',"slice","slice","slice"],
                'Direction': ['Crosscourt', 'Down Middle', 'Down the Line', 'Crosscourt', 'Down Middle', 'Down the Line','Crosscourt', 'Down Middle', 'Down the Line'],
                'Shots': p1}
    df_dblepieP1 = pd.DataFrame(data)

    # créer un diagramme à secteurs imbriqués
    fig1 = px.sunburst(df_dblepieP1, path=['Direction','Shot Type'], values='Shots')
    col1.plotly_chart(fig1, use_container_width=True)

    p2 = doublepiechart(df_joueur, player2)
    data = {'Shot Type': ['Forehand', 'Forehand', 'Forehand', 'Backhand', 'Backhand', 'Backhand',"slice","slice","slice"],
                'Direction': ['Crosscourt', 'Down Middle', 'Down the Line', 'Crosscourt', 'Down Middle', 'Down the Line','Crosscourt', 'Down Middle', 'Down the Line'],
                'Shots': p2}
    df_dblepieP2 = pd.DataFrame(data)

    # créer un diagramme à secteurs imbriqués
    fig2 = px.sunburst(df_dblepieP2, path=['Direction','Shot Type'], values='Shots')
    col2.plotly_chart(fig2, use_container_width=True)

        
#     df_serve_basics = pd.read_csv(path+ 'df_serve_basics.csv')
#     def moy_var_j(data,joueur,variable):      #ICI LA VARIABLE POUR FAIRE RESSORTIR LA MOYENNE D'UNE VARIABLE D'UN JOUEUR
#         data= (data[(data['j1'] ==joueur) | (data['j2'] == joueur)])  
#         moyenne= data[variable].mean()
#         return moyenne
    
#     ace_j1= moy_var_j(df_serve_basics, player1, "aces")
#     ace_j2= moy_var_j(df_serve_basics, player2, "aces")          #j'affecte ce que je veux à des variables
#     unret_j1= moy_var_j(df_serve_basics, player1, "unret")
#     unret_j2= moy_var_j(df_serve_basics, player2, "unret")
    
#     J1= [ace_j1,unret_j1]           #Je liste mes variables pour le radar
#     J2=[ace_j2,unret_j2]
#     print("=>",J1,J2) # print de vérif
    
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(r=J1,theta= ['aces','unret'],fill='toself', name=f'Aces et Unret {player1}',))
#     fig.add_trace(go.Scatterpolar(r=J2,theta=['aces','unret'],fill='toself',name=F'Aces et Unret {player2}', line=dict(color='rgba(255, 0, 0, 0.5)')))

#     fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 50])),showlegend=True    )

#     st.plotly_chart(fig)
    
    
    
    
    
    #####################################################  MACHINE LEARNING ##############################################################  ######################################################################################################################################


    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    # Chargement du dataset contenant les features des player1s
    
    st.markdown("<h1 style='text-align: center; color : #ac211e'>AND NOW, LADIES AND GENTLEMEN...</h1>", unsafe_allow_html=True)
    
    choix_surfaces = st.selectbox("", options = ['Choisissez une surface', 'Toutes', 'Clay', 'Grass', 'Hard'], key="C")
    
    lencoder = LabelEncoder()

    df['surface_enc'] = lencoder.fit_transform(df['surface'])
    df['tourney_level_enc'] = lencoder.fit_transform(df['tourney_level'])

    if choix_surfaces != 'Choisissez une surface' :
        if choix_surfaces == 'Toutes':
            X = df[['player_1_surface_winrate', 'player_1_svpt', 'player_1_bpFaced', 'player_2_surface_winrate',  'player_2_svpt',  'player_2_bpFaced']]
            y = df['target']
        else :    
            X = df[df['surface'] == choix_surfaces][['player_1_surface_winrate', 'player_1_svpt', 'player_1_bpFaced', 'player_2_surface_winrate',  'player_2_svpt',  'player_2_bpFaced']]
            y = df[df['surface'] == choix_surfaces]['target']
        # bpFaced : number of break points faced
        # svpt : number of serve points

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle= True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # création du modèle
        model = LogisticRegression(C = 0.1, max_iter = 100, penalty= 'l1', solver= 'liblinear').fit(X_train,y_train)




        # MATCH !
              
        col1, col2, col3 = st.columns([2,3,2])

        bouton = col1.button("MATCH !")
        
        col2.write("")
        col2.write("")

        if bouton :

            moy_df= pd.read_csv(path +"DFMOYV5b.csv", usecols = ['Joueur', 'svpt', 'bpFaced', 'Toutes', 'Clay', 'Grass', 'Hard'])
            moy_df_cols_num = moy_df[moy_df.select_dtypes("number").columns]
            
            try :
                if choix_surfaces != 'Toutes':
                    duels_surface = duels[duels['surface'] == choix_surfaces]
                else:
                    duels_surface = duels
                    
                vJ1 = len(duels_surface[duels_surface['winner'] == player1])
                vJ2 = len(duels_surface[duels_surface['winner'] == player2])


                taux_vic_J1_matchs_h2h = vJ1/len(duels_surface)
                taux_vic_J2_matchs_h2h = vJ2/len(duels_surface)
        
            except :
                taux_vic_J1_matchs_h2h = 1
                taux_vic_J2_matchs_h2h = 1
                
            st.write(taux_vic_J1_matchs_h2h)
            st.write(taux_vic_J2_matchs_h2h)
            
            # changement échelle du DF

            scaler = StandardScaler()
            moy_df_cols_num_scaled = scaler.fit_transform(moy_df_cols_num)
            moy_df_cols_num_scaled = pd.DataFrame(moy_df_cols_num_scaled, columns=moy_df_cols_num.columns)
            merged_df = pd.merge(moy_df[['Joueur']], moy_df_cols_num_scaled, left_index=True, right_index=True)

            # selection features
            
            player1_features = merged_df[merged_df['Joueur'] == player1][[choix_surfaces,'svpt','bpFaced']]
            player2_features = merged_df[merged_df['Joueur'] == player2][[choix_surfaces,'svpt','bpFaced']]
            
            st.write(player1_features)
            st.write(player2_features)
            
            player1_features[[choix_surfaces,'svpt','bpFaced']] = player1_features[[choix_surfaces,'svpt','bpFaced']] * taux_vic_J1_matchs_h2h
            player2_features[[choix_surfaces,'svpt','bpFaced']] = player2_features[[choix_surfaces,'svpt','bpFaced']] * taux_vic_J2_matchs_h2h

            st.write(player1_features)
            st.write(player2_features)
            
            player1_array = player1_features.iloc[0].to_numpy()
            player2_array = player2_features.iloc[0].to_numpy()

            concatenated_array = np.concatenate([player1_array, player2_array])
            winner = model.predict(concatenated_array.reshape(1, -1))


            # Affichage du résultat
                                   
            if winner == 'player_1' :
                col2.success(f"Le gagnant est {player1}")
                try:
                    imagej1 = Image.open(path + "\\photos\\" + player1.split(' ')[1].lower()+".png")
                    overlay = Image.open(path + "\\photos\\" + "coupe.png")

                    # Convertir en mode RGBA pour préserver la transparence
                    imagej1 = imagej1.convert("RGBA")
                    overlay = overlay.convert("RGBA")

                    # Combinez les deux images avec transparence
                    new_img = Image.alpha_composite(imagej1, overlay)
                    new_img.save("new.png","PNG")                   
                    col2.image(new_img)

                except:
                    imagej1 = Image.open(path + "\\photos\\" + "logo2.png")
                    overlay = Image.open(path + "\\photos\\" + "coupe.png")
                    imagej1 = imagej1.convert("RGBA")
                    overlay = overlay.convert("RGBA")
                    new_img = Image.alpha_composite(imagej1, overlay)
                    new_img.save("new.png","PNG")
                    col2.image(new_img)

            else :
                col2.success(f"Le gagnant est {player2}")
                try:
                    imagej2 = Image.open(path + "\\photos\\" + player2.split(' ')[1].lower()+".png")
                    overlay = Image.open(path + "\\photos\\" + "coupe.png")
                    imagej2 = imagej2.convert("RGBA")
                    overlay = overlay.convert("RGBA")
                    new_img = Image.alpha_composite(imagej2, overlay)
                    new_img.save("new.png","PNG")                   
                    col2.image(new_img)
                except:
                    imagej2 = Image.open(path + "\\photos\\" + "logo2.png")
                    overlay = Image.open(path + "\\photos\\" + "coupe.png")
                    imagej2 = imagej2.convert("RGBA")
                    overlay = overlay.convert("RGBA")
                    new_img = Image.alpha_composite(imagej2, overlay)
                    new_img.save("new.png","PNG")
                    col2.image(new_img)