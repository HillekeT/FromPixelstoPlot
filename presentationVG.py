import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

from PIL import Image
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

# charger bases des données
vgsales = pd.read_csv('vgsales.csv')
selection_vg = pd.read_csv('df_kaggle1.csv')
avis_series = pd.read_csv('Avis_series.csv')
notes_FF = pd.read_csv('game_notes_dataFF.csv')
notes_TR = pd.read_csv('game_notes_dataTR.csv')
notes_DN = pd.read_csv('game_notes_dataDN.csv')

# Remplacer les éditeurs 'Square', 'Square Enix', 'Square EA', et 'Square Soft' par 'SquareGroup'
selection_vg['Publisher'] = selection_vg['Publisher'].replace(['Square', 'Square Enix', 'Square EA', 'SquareSoft'], 'SquareGroup')

# Nettoyer les données et regrouper par série
df_cleaned = selection_vg.groupby('Name').agg({
    'Platform': 'first',
    'Year': 'first',
    'Genre': 'first',
    'Publisher': 'first',
    'NA_Sales': 'sum',
    'EU_Sales': 'sum',
    'JP_Sales': 'sum',
    'Other_Sales': 'sum',
    'Global_Sales': 'sum'
}).reset_index()

# Supprimer les lignes avec Global_Sales nul
df_cleaned = df_cleaned[df_cleaned['Global_Sales'].notna()]

# Ajouter une colonne pour la série
def classify_series(name):
    if 'Final Fantasy' in name:
        return 'Final Fantasy'
    elif 'Tomb Raider' in name:
        return 'Tomb Raider'
    elif 'Duke Nukem' in name:
        return 'Duke Nukem'
    else:
        return 'Unknown'

df_cleaned['Series'] = df_cleaned['Name'].apply(classify_series)

# Couleurs pour chaque série
series_colors = {
    'Final Fantasy': 'blue',
    'Tomb Raider': 'green',
    'Duke Nukem': 'red'
}

# Fonction pour tracer les ventes globales avec deux axes y (ventes globales et nombre de jeux)
def plot_sales_and_games(series_name, color):
    series_data = df_cleaned[df_cleaned['Name'].str.contains(series_name, case=False)]
    sales_by_year = series_data.groupby('Year').agg({
        'Global_Sales': 'sum',
        'Name': 'count'  # Nombre de jeux par année
    }).reset_index()

    fig = go.Figure()

    # Ajouter les ventes globales
    fig.add_trace(go.Scatter(
        x=sales_by_year['Year'],
        y=sales_by_year['Global_Sales'],
        mode='markers',
        marker=dict(color=color, size=10),
        name='Ventes Globales'
    ))

    # Ajouter le nombre de jeux (2ème axe y)
    fig.add_trace(go.Scatter(
        x=sales_by_year['Year'],
        y=sales_by_year['Name'],
        mode='lines+markers',
        line=dict(color='orange'),
        marker=dict(size=8),
        name='Nombre de Jeux',
        yaxis='y2'
    ))

    # Mise en page avec les deux axes y et déplacement de la légende
    fig.update_layout(
        title=f'Ventes Globales et Nombre de Jeux par Année pour {series_name}',
        xaxis_title='Année',
        yaxis=dict(
            title='Ventes Globales (en millions)',
            titlefont=dict(color=color),
            tickfont=dict(color=color)
        ),
        yaxis2=dict(
            title='Nombre de Jeux',
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.8, y=1.2,  # Ajuste la position de la légende
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1
        ),
        template='plotly_white'
    )

    return fig

# Fonction pour tracer les ventes globales uniquement
def plot_global_sales(series_name, color):
    series_data = df_cleaned[df_cleaned['Name'].str.contains(series_name, case=False)]
    sales_by_year = series_data.groupby('Year').agg({
        'Global_Sales': 'sum'
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sales_by_year['Year'],
        y=sales_by_year['Global_Sales'],
        mode='lines+markers',
        name=series_name,
        line=dict(color=color),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f'Ventes Globales pour {series_name} par Année',
        xaxis_title='Année',
        yaxis_title='Ventes Globales (en millions)',
        template='plotly_white',
        legend=dict(
            x=0.8, y=1.2,  # Ajuste la position de la légende
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1
        )
    )

    return fig

# Fonction pour créer un graphique des ventes par région
def plot_sales_by_region(series_name, color):
    series_data = df_cleaned[df_cleaned['Series'] == series_name]
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    region_labels = ['North America', 'Europe', 'Japan', 'Other']
    region_colors = ['blue', 'green', 'red', 'orange']
    
    fig = go.Figure()

    for region, label, reg_color in zip(regions, region_labels, region_colors):
        sales_by_year = series_data.groupby('Year').agg({region: 'sum'}).reset_index()
        fig.add_trace(go.Scatter(
            x=sales_by_year['Year'],
            y=sales_by_year[region],
            mode='lines+markers',
            name=label,
            line=dict(color=reg_color),
            marker=dict(color=reg_color)
        ))
    
    fig.update_layout(
        title=f'Ventes Régionales pour {series_name} par Année',
        xaxis_title='Année',
        yaxis_title='Ventes (en millions)',
        template='plotly_white',
        height=600,  # Agrandir les graphes
        width=1000,
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='white'
    )
    return fig

# Fonction pour tracer les ventes globales avec annotations
def plot_sales_over_time_with_annotations(df, colors):
    fig = go.Figure()

    # Tracer chaque série
    for series, color in colors.items():
        df_series = df[df['Series'] == series]
        df_series = df_series.groupby('Year')['Global_Sales'].sum().reset_index()
        
        fig.add_trace(go.Scatter(
            x=df_series['Year'], 
            y=df_series['Global_Sales'], 
            mode='lines+markers', 
            name=series, 
            line=dict(color=color)
        ))

    # Ajouter des annotations ajustées pour les événements spécifiques
    annotations = [
        dict(
            x=1997, y=13, xref="x", yref="y",
            text="Sortie de Final Fantasy VII",
            showarrow=True,
            arrowhead=2,
            ax=-20, ay=-70,
            font=dict(color="blue", size=10),
            bgcolor="white",
            bordercolor="blue"
        ),
        dict(
            x=1996, y=5.7, xref="x", yref="y",
            text="Sortie de Tomb Raider",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-70,
            font=dict(color="green", size=10),
            bgcolor="white",
            bordercolor="green"
        ),
        dict(
            x=2001, y=8.6, xref="x", yref="y",
            text="Sortie de la PlayStation 2",
            showarrow=True,
            arrowhead=2,
            ax=-10, ay=-70,
            font=dict(color="purple", size=10),
            bgcolor="white",
            bordercolor="purple"
        ),
        dict(
            x=1999, y=9.19, xref="x", yref="y",
            text="Sortie de Final Fantasy VIII",
            showarrow=True,
            arrowhead=2,
            ax=10, ay=-85,
            font=dict(color="blue", size=10),
            bgcolor="white",
            bordercolor="blue"
        ),
        dict(
            x=2013, y=6.8, xref="x", yref="y",
            text="Reboot de Tomb Raider",
            showarrow=True,
            arrowhead=2,
            ax=0, ay=-50,
            font=dict(color="green", size=10),
            bgcolor="white",
            bordercolor="green"
        ), 
        dict(
            x=2009, y=9.6, xref="x", yref="y",
            text="Sortie de Final Fantasy XIII",
            showarrow=True,
            arrowhead=2,
            ax=-20, ay=-80,
            font=dict(color="blue", size=10),
            bgcolor="white",
            bordercolor="blue"
        )
    ]

    fig.update_layout(
        title='Évolution des ventes globales par année',
        xaxis_title='Année',
        yaxis_title='Ventes globales (en millions)',
        annotations=annotations,
        width=1200,
        height=600
    )
    
    return fig

# Fonction pour tracer les ventes par région pour toutes les séries avec Plotly
def plot_all_series_regional_sales_plotly(df, colors):
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    region_names = ['Amérique du Nord', 'Europe', 'Japon', 'Autres']

    # Préparer les données
    regional_sales = df.groupby('Series')[regions].sum().T

    # Créer le graphique Plotly
    fig = go.Figure()

    # Ajouter les traces pour chaque série
    for series in regional_sales.columns:
        fig.add_trace(go.Bar(
            x=region_names,
            y=regional_sales[series],
            name=series,
            marker=dict(color=colors[series])
        ))

    # Mettre à jour la mise en page du graphique
    fig.update_layout(
        title="Comparaison des ventes par région pour toutes les séries",
        xaxis_title="Région",
        yaxis_title="Ventes (en millions)",
        barmode='group',
        legend_title="Série",
        template='plotly_white'
    )

    return fig

def clean_numeric_column(column):
    return pd.to_numeric(column.replace({'K': '', 'Null': None}, regex=True), errors='coerce') * 1000

frames = [notes_FF, notes_TR, notes_DN]
notes_series = pd.concat(frames, ignore_index=True)

notes_series['Etoiles'] = clean_numeric_column(notes_series['Etoiles'])
notes_series['Envie_de_jouer'] = clean_numeric_column(notes_series['Envie_de_jouer'])
notes_series['Coup_de_coeur'] = clean_numeric_column(notes_series['Coup_de_coeur'])
notes_series['Note'] = pd.to_numeric(notes_series['Note'], errors='coerce')

notes_series['Année'] = pd.to_numeric(notes_series['Année'], errors='coerce').astype('Int64')  # Utilisation de 'Int64' pour les valeurs manquantes
df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce').astype('Int64')

# Conserver la note la plus élevée pour chaque jeu avec le même nom et la même année
notes_series = notes_series.groupby(['Titre', 'Année']).agg({
    'Note': 'max',
    'Etoiles': 'max',
    'Envie_de_jouer': 'max',
    'Coup_de_coeur': 'max'
}).reset_index()

# Fusionner les données de ventes et de notes
df_combined = pd.merge(df_cleaned, notes_series, left_on='Name', right_on='Titre', how='inner')








df_cloud = pd.read_csv('Avis_series_SA2.csv')

df_DN_hist = df_cloud.loc[df_cloud.jeu_serie == 'DN']
df_FF_hist = df_cloud.loc[df_cloud.jeu_serie.isin(['FF', 'FF6', 'FF7'])]
df_TR_hist = df_cloud.loc[df_cloud.jeu_serie.isin(['TR', 'TR2'])]

color_map = {
        'POSITIF': 'green',
        'NEGATIF': 'red',
        'NEUTRAL': 'orange'
    }

def create_histogram(df, title):
    fig = px.histogram(df, 
                       x='serie', 
                       color='type', 
                       color_discrete_map=color_map,
                       title=title)
    fig.update_layout(yaxis=dict(title="Nb d'avis"))  
    return fig

# préparation pour faire les nuages des mots
stopcorpus = set(stopwords.words('french'))
additional_stopwords = ["?", ".", ",", ":", ";", "...", "(", ")" "'", "-", "!", "a", "ff", "va", "rien", "qte", 
                        "assez", "point", "parce", "fois", "quelques", "dire", "deux", "an", "plus", "peu", "après", "sans", "trop", 
                        "vraiment", "non", "donc", "très", "où", "là", "quand", "c'est", "comme", "tout","duke", "nukem", "tomb", 
                        "raider", "alors", "encore", "si", "vi", "vii", "final", "fantasy", "être", "avoir", "peut", "aussi", "fait", 
                        "le", "la", "les", "de", "du", "des", "et", "un", "une", "en", "à", "avec", "pour", "par", "est", "qui", "que", "sur", 
                        "dans", "ce", "cette", "il", "elle", "nous", "vous", "ils", "elles", "c'est", "ça", "j'ai", "aux", "mais", "pas", "car", 
                        "maintenant", "jeu", "pa", "faire", "ou", "certain", "certains", "autres", "tous", "beaucoup", "6", "VI", "VII", "version", 
                       "jeux", "titre", "cela", "autre", "bien", "reste", "toujours", "beaucoup", "déjà", "bon", "moins", "ff7", "ffvi", "surtout",
                        "jamais", "entre", "plutôt", "partie", "est", "2", "oui", "ffvii", "puis", "ailleurs", "quoi", "enfin", "malgré", "pendant"]
stopcorpus.update(additional_stopwords)

filtered_data_dn = df_cloud.loc[df_cloud.jeu_serie.isin(['DN'])]
filtered_data_ff6 = df_cloud.loc[df_cloud.jeu_serie.isin(['FF6'])]
filtered_data_ff7 = df_cloud.loc[df_cloud.jeu_serie.isin(['FF7'])]
filtered_data_tr = df_cloud.loc[df_cloud.jeu_serie.isin(['TR', 'TR2'])]

def preprocess_text(data, stopcorpus):
    def style_text(text:str):
        return text.lower()

    def remove_words(words, stopwords):
        return [word for word in words if word.lower() not in stopwords]

    def collapse_list_to_string(words):
        return " ".join(words)

    def remove_apostrophes(text):
        text = text.replace('"', "")
        text = text.replace('`', "")
        text = text.replace(',', "")
        text = text.replace('.', " ")
        text = text.replace("'", " ")
        text = text.replace("«", "")
        text = text.replace("»", "")
        text = text.replace("!", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        return text

    # Appliquez les transformations
    data['cleaned_text'] = data['Content'].astype(str)
    data['cleaned_text'] = data['cleaned_text'].astype(str).apply(style_text)
    data['cleaned_text'] = data['cleaned_text'].apply(remove_apostrophes)
    data['cleaned_text'] = data['cleaned_text'].astype(str).apply(lambda x: remove_words(x.split(), stopcorpus))
    data['cleaned_text'] = data['cleaned_text'].apply(collapse_list_to_string)

    return data

data_DN = preprocess_text(filtered_data_dn, stopcorpus)
data_FF6 = preprocess_text(filtered_data_ff6, stopcorpus)
data_FF7 = preprocess_text(filtered_data_ff7, stopcorpus)
data_TR = preprocess_text(filtered_data_tr, stopcorpus)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def collapse_list_to_string(words):
    return " ".join(words)

def lemm_data(data): 
    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    # Appliquez les transformations
    data.loc[:, 'clean_lemmatized'] = data['cleaned_text'].astype(str).apply(lemmatize_text)
    data.loc[:, 'clean_lemmatized'] = data['clean_lemmatized'].apply(collapse_list_to_string)

    return data

data_DN = lemm_data(data_DN)
data_FF6 = lemm_data(data_FF6)
data_FF7 = lemm_data(data_FF7)
data_TR = lemm_data(data_TR)

def plot_wordcloud(series, masque):
    # Chargement du masque
    masque = np.array(Image.open(str(masque)))

    # Génération des couleurs à partir de l'image du masque
    img_color = ImageColorGenerator(masque)
    
    wordcloud = WordCloud(background_color='white', 
                          max_words=250,
                          max_font_size=120,
                          collocations=False,
                          random_state=42,
                          mask=masque,  # Utilisation du masque pour la forme du nuage
                          color_func=img_color
                        ).generate(' '.join(series.astype(str)))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def plot_most_frequent_words(data, sentiment, text_column, serie, top_n):
    data_sentiment = data.loc[data['type'] == sentiment]
    
    chaine = ' '.join(i.lower() for i in data_sentiment[text_column])

    # Compte l'occurrence des mots
    dico = Counter(chaine.split())

    # Obtenir les mots les plus fréquents et leur fréquence
    mots = [m[0] for m in dico.most_common(top_n)]
    freq = [m[1] for m in dico.most_common(top_n)]

    # Créer le graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mots, y=freq)
    plt.title(f'{top_n} mots les plus fréquemment employés dans les critiques "{sentiment.capitalize()}" de {serie}')
    plt.xlabel('Mots')
    plt.ylabel('Fréquence')
    plt.xticks(rotation=45)  # Rotation des étiquettes pour une meilleure lisibilité
    return plt

# Defining the various content display functions
def display_introduction():
    st.markdown("<h2 style='text-align: center; color: white;'>Introduction</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Projet réalisé dans le cadre de la formation Data Analyst (Bootcamp juin 2024) de Datascientest</h6>", unsafe_allow_html=True)

    groupe = st.container()
    objectif = st.container()
    context = st.container()

    with groupe:
        st.subheader("Groupe")
        col1, col2, col3, col4 = st.columns([0.25, 1, 0.25, 1])

        with col1:
            st.image("HT.jpg", width=70)
        with col2:
            st.markdown("**Hildegarde THYBERGHIEN**")
            st.markdown("https://www.linkedin.com/in/hildegarde-thyberghien-553623317")

        with col3:
            st.image("Alexpass.jpg", width=70)
        with col4:
            st.markdown("**Alexandre LAZERAT**")
            st.markdown("https://www.linkedin.com/in/lazerat/")

        with col1:
            st.image("Charafpass.jpg", width=70)
        with col2:
            st.markdown("**Charaf Eddine BEJJIT**")
            st.markdown("https://www.linkedin.com/in/charaf-eddine-bejjit-805304168/")

        with col3:
            st.image("Léopass.jpg", width=70)
        with col4:
            st.markdown("**Léo CADART**")
            st.markdown("https://www.linkedin.com/in/leo-cadart-8475871aa/")

    with objectif:
        st.subheader("Projet Objectif")
        st.markdown("""Le projet vise à identifier les facteurs de succès des ventes des séries de jeux vidéo **‘Final Fantasy’**, **‘Tomb Raider’** et
        **‘Duke Nukem’**, et à établir des comparaisons entre ces franchises.""")

        st.markdown("""Ces choix se basent sur leur importance dans l'industrie, chacune ayant marqué son époque et attiré de nombreux joueurs.
        ‘Final Fantasy’ se distingue par son **innovation et ses récits immersifs**, ‘Tomb Raider’ a redéfini **le genre d'aventure avec sa 
        protagoniste emblématique**, tandis que ‘Duke Nukem’ illustre **les évolutions et défis** de certaines franchises. L'analyse offre des 
        perspectives sur les succès individuels et collectifs, ainsi que sur les tendances applicables à l'ensemble de l'industrie du jeu vidéo.""")

    with context:
        col5, col6 = st.columns(2)
    
        col5.subheader("Context")
        col5.markdown("""L'utilisation des données dans l'industrie du jeu vidéo est devenue cruciale, surtout avec la montée de la distribution 
        numérique et des jeux en ligne. Les entreprises peuvent désormais exploiter d'importantes quantités de données sur les comportements 
        et préférences des joueurs, ce qui leur permet d'analyser l'engagement, d'identifier des tendances et de développer des stratégies 
        marketing ciblées. De plus, cette analyse des données peut influencer les décisions de conception des jeux, conduisant à des productions
        plus attractives et réussies.""")
    
        col5.markdown("""Une remarque importante à noter est que nous n'avons pas trouvé de données sur les ventes après 2016. En effet, la part de marché numérique des logiciels,
        en particulier des jeux vidéo, est très élevée, mais souvent peu accessible au public. Le dataset proposé par VGChartz, qui se concentre uniquement sur les
        ventes physiques, limite ainsi la fiabilité des estimations de ventes au détail après 2016, rendant ces chiffres de moins en moins représentatifs de la 
        performance globale des jeux concernés.""")

        col6.image("sceneFF7.jpg", caption="scène d'ouverture Final Fantasy VII")


def display_data():
    st.title("Données")
    st.subheader("Les données sur les ventes de jeux vidéo")
    st.markdown("""Nous avons d’abord utilisé une base de données provenant du site **Kaggle** ([lien](https://www.kaggle.com/datasets/gregorut/videogamesales)). Elle a été générée par un extraction de vgchartz.com. La base de données 
    contient une liste de plus de 16 500 jeux vidéo ayant des ventes supérieures à 100 000 exemplaires et contient des informations sur les plateformes de jeu, le genre, l’éditeur,
    ainsi que les ventes par région : Global, Amérique du Nord, Europe, Japon et Autres.""")

    st.markdown("""Le jeu de données initial, vgsales.csv, présente une volumétrie de 11 colonnes et 16 598 lignes, soit un total de 11 493 valeurs.""")
    st.markdown("""Période : 1987 - 2016""")

    st.markdown('<span style="color: #F2AA84; font-weight: bold;">Visualiser la base de données du site Kaggle</span>', unsafe_allow_html=True)
    if st.checkbox("Afficher la base de données", key="show_dataframe_vgsales"):
        st.dataframe(vgsales)
    
    st.markdown("""
    **Veuillez prendre en compte les points suivants concernant la base de données :** 📊
    """)
    st.markdown("""
    - **Données Cumulatives** : Les ventes sont des données cumulatives jusqu’en 2016. Par conséquent, les jeux les plus récents ont des ventes sous-représentées. 
      De plus, comme indiqué dans un paragraphe précédent, le site ne publie plus de chiffres après 2018 car ils ne sont plus représentatifs.

    - **Plateforme** : Les ventes sont représentées par plateforme d’un jeu.

    - **Jeux Exclus** : Dans ce jeu de données, les jeux mobiles et les jeux indépendants ne sont pas inclus.

    - **Téléchargements Gratuits** : Les jeux avec des téléchargements gratuits ne sont pas pris en compte dans les ventes, ce qui exclut des titres à succès tels que 
      ‘Counter-Strike’ et ‘League of Legends’.
    """)

    st.markdown("""
    L’objectif de notre projet était de comparer certains jeux entre eux. Nous avons donc extrait les données des trois séries de jeu choisis :
    - Final Fantasy 
    - Tomb Raider
    - Duke Nukem
    """)

    st.markdown('<span style="color: #F2AA84; font-weight: bold;">Visualiser la base de données avec la séléction de séries</span>', unsafe_allow_html=True)
    if st.checkbox("Afficher la base de données", key="show_dataframe_selection"):
        st.dataframe(selection_vg)

    st.subheader("Les données manquantes sur les jeux vidéo")
    st.markdown("""Pour compléter les données de ventes, nous avons utilisé de **webscraping** pour récupérer différentes notes d’appréciation, commentaires et autres informations sur le site de **SensCritique** ([lien](https://www.senscritique.com/jeuvideo)). Une première base de données contient des informations sur les notes moyennes, le nombre d’étoiles, le nombre de coups de cœur et les envies de jouer. Une deuxième base de données regroupe les notes et les commentaires laissés par des utilisateurs sur un jeu particulier.
    """)
    st.markdown("""En tout, 6 bases de données ont été utilisées pour les analyses.""")

    st.markdown('<span style="color: #F2AA84; font-weight: bold;">Afficher les bases des données</span>', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.checkbox("Avis Séries", key="show_dataframe_avis"):
            st.dataframe(avis_series)
    with colB:
        if st.checkbox("Notes Final Fantasy", key="show_dataframe_notesFF"):
            st.dataframe(notes_FF)
    with colC:
        if st.checkbox("Notes Tomb Raider", key="show_dataframe_notesTR"):
            st.dataframe(notes_TR)
    with colD:
        if st.checkbox("Notes Duke Nukem", key="show_dataframe_notesDN"):
            st.dataframe(notes_DN)

    st.subheader("La préparation des bases de données")
    st.markdown("""
    À partir des bases de données contenant les notes moyennes et les chiffres de ventes, nous avons entrepris un processus rigoureux de nettoyage des données, comprenant
    les étapes suivantes :   
    - **Suppression des Valeurs Nulles :** Nous avons éliminé toutes les lignes contenant des valeurs nulles, afin d'assurer l'intégrité des données et d'éviter des biais dans les analyses ultérieures.
    - **Vérification et Suppression des Doublons :** Une revérification minutieuse a révélé la présence de doublons dans le jeu de données. Nous avons donc supprimé ces entrées redondantes pour ne conserver que les enregistrements uniques. De plus, il est important de noter que certains jeux, comme Tomb Raider, étaient enregistrés sous le même nom mais avaient été publiés à des années différentes, ce qui a nécessité une clarification dans la présentation de ces données pour éviter toute confusion.
    - **Conversion des Types de Données :** Nous avons identifié des variables présentant des formats incompatibles. Par exemple, les ventes et les notes étaient initialement stockées sous forme de texte. Ces valeurs ont été converties aux types numériques appropriés pour garantir leur utilisation correcte lors des analyses et des opérations de fusion des jeux de données.
    - **Fusion des Données :** Les jeux de données concernant les ventes et les notes ont été fusionnés en utilisant des clés communes, telles que le titre du jeu et l'année de sortie. Cette étape a permis d'intégrer les informations pertinentes et de faciliter les analyses croisées entre les performances commerciales et les évaluations critiques.
    """)

def display_visualisation():
    st.title("Exploration & Visualisation")
    st.subheader('Analyse du nombre de jeux par série')

    colE, colF = st.columns(2)
    with colE :
        # Graphique nombre de jeux par série
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_cleaned, x='Series', order=df_cleaned['Series'].value_counts().index, palette=series_colors)
        plt.xticks(rotation=45)
        plt.title('Nombre de jeux par série')
        plt.xlabel('Série')
        plt.ylabel('Nombre de jeux')
        st.pyplot(plt)

    with colF:
        st.write("""
        Le graphique montre que le nombre de jeux publiés varie considérablement entre les séries. Cette différence devra être prise en compte dans l'analyse des résultats, car elle peut influencer la comparaison des performances des ventes globales.
        """)

    st.subheader("Analyse des ventes par année pour chaque série")
    series_list = ['Final Fantasy', 'Tomb Raider', 'Duke Nukem']
    selected_series = st.selectbox("Choisissez une série à analyser:", series_list)
    color = series_colors.get(selected_series, 'blue')

    # Afficher les graphiques côte à côte
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_global_sales(selected_series, color))

    with col2:
        st.plotly_chart(plot_sales_and_games(selected_series, color))

    # Commentaire spécifique pour chaque série
    if selected_series == 'Final Fantasy':
        st.write("""
        La série **Final Fantasy** est un leader dans l'industrie des jeux vidéo, avec des ventes globales atteignant des sommets historiques grâce à des titres comme *Final Fantasy VII*. Après une période de baisse dans les années 2000, des jeux comme *Final Fantasy X* et *Final Fantasy XIII* ont permis de maintenir l'intérêt des joueurs. Le lancement des nouvelles générations de consoles a souvent contribué à la relance des ventes.
        """)
    elif selected_series == 'Tomb Raider':
        st.write("""
        La série **Tomb Raider** a connu un pic de ventes en 1996 lors de son lancement, suivi d'une baisse dans les années 2000. Cependant, le reboot en 2013 a relancé la franchise, attirant une nouvelle génération de joueurs. Le personnage emblématique de Lara Croft et les rééditions ont joué un rôle clé dans cette reprise.
        """)
    elif selected_series == 'Duke Nukem':
        st.write("""
        La série **Duke Nukem** a vu ses ventes stagner après 2000, en raison de l'absence de nouvelles sorties. En 2011, un retour nostalgique a relancé les ventes, mais l'impact est resté limité comparé à d'autres grandes franchises. Duke Nukem reste une série culte, appréciée par une base de fans fidèle.
        """)

    # Conclusion globale pour toutes les séries
    st.markdown("""
    <span style="color: #F2AA84; font-weight: bold;">Les trois séries analysées présentent des tendances de ventes différentes. **Final Fantasy** domine le marché avec des sorties régulières et des pics notables. **Tomb Raider** a connu un pic initial suivi d'une baisse, mais le reboot de 2013 a permis de relancer la franchise. **Duke Nukem**, bien que culte, a eu un impact plus modeste, avec un pic de vente en 2013 principalement due à la nostalgie des fans.</span>
    """, unsafe_allow_html=True)

    st.subheader('Ventes régionales par année pour chaque série')
    series_list = df_cleaned['Series'].unique().tolist()
    selected_series = st.selectbox("Choisissez une série à analyser", series_list)
    color_map = {'Final Fantasy': 'blue', 'Tomb Raider': 'green', 'Duke Nukem': 'red'}
    color = color_map.get(selected_series, 'blue')

    st.plotly_chart(plot_sales_by_region(selected_series, color), use_container_width=True)

    # Résumé court basé sur les séries
    if selected_series == "Final Fantasy":
        st.write("""
        ##### Final Fantasy
        - **Prédominance du Japon** : Dominante des ventes au Japon dans les années 90 grâce à une base de fans solide.
        - **Croissance en Amérique du Nord et Europe** : Les ventes en Occident ont augmenté à partir de la fin des années 90 grâce à des titres comme *Final Fantasy VII*.
        - **Régionalité des succès** : Bien que le Japon ait été fort au début, l'Amérique du Nord et l'Europe ont rattrapé leur retard après 2000.
        """)
    elif selected_series == "Tomb Raider":
        st.write("""
        ##### Tomb Raider
        - **Équilibre sur le marché occidental** : Ventes bien réparties entre l'Europe et l'Amérique du Nord.
        - **Ventes modestes au Japon** : Tomb Raider n'a jamais eu le même succès au Japon que Final Fantasy en raison de différences culturelles.
        - **Stabilité des ventes globales** : Des pics en 1997 et 2013, mais une stabilité relative dans le reste des années.
        """)
    elif selected_series == "Duke Nukem":
        st.write("""
        ##### Duke Nukem
        - **Concentration sur les marchés occidentaux** : Le jeu est surtout populaire en Amérique du Nord et en Europe, avec peu d'impact au Japon.
        - **Absence de nouveaux titres** : Un long creux entre 2000 et 2010, suivi d'un retour modeste en 2011 grâce à la nostalgie.
        """)

    # Bilan global
    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Final Fantasy a connu un succès mondial, avec des ventes initialement dominées par le marché japonais, mais a progressivement élargi son influence en Amérique du Nord et en Europe à partir des années 2000.    </span>""", unsafe_allow_html=True)
    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Tomb Raider a connue une grande popularité dans l'Europe et l'Amérique du Nord, mais a toujours été moins populaire au Japon.</span>""", unsafe_allow_html=True)
    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Duke Nukem, bien qu'une série culte en Occident, a eu peu d'impact sur le marché oriental. Mais elle reste une série avec des ventes concentrées sur des périodes spécifiques et limitées.</span>""", unsafe_allow_html=True)

    st.subheader("Comparaison des ventes entre Final Fantasy, Tomb Raider, et Duke Nukem")

    fig = plot_sales_over_time_with_annotations(df_cleaned, series_colors)
    with st.container():
        st.plotly_chart(fig)

    st.write("""
    Les graphes ci-dessous présentent une vue d'ensemble de l'évolution des ventes globales des trois séries emblématiques : Final Fantasy, Tomb Raider, et Duke Nukem. Cette comparaison permet de mettre en lumière les différences de trajectoires entre ces franchises, ainsi que l'impact de certains événements majeurs, comme la sortie de nouveaux titres ou de consoles.

    #####  Final Fantasy :
    1. **Période de Pic (1997-2000)** :
        - 1997 est marqué par la sortie de *Final Fantasy VII*, qui est le jeu le plus vendu de la franchise, atteignant près de 10 millions de ventes globales sur PlayStation. Ce jeu a révolutionné le genre RPG et a introduit *Final Fantasy* à un public mondial beaucoup plus large.
        - En 1999, *Final Fantasy VIII* a également connu un grand succès, maintenant les ventes globales à un niveau élevé avant le déclin du début des années 2000.

    2. **Déclin et Fluctuations Post-2000** :
        - Après 2000, les ventes de *Final Fantasy* connaissent un déclin, notamment en 2005 avec *Final Fantasy IV Advance*, qui n'a pas relancé la série comme les titres précédents.
        - Le retour de la franchise en 2009 avec *Final Fantasy XIII* a capitalisé sur la popularité de la PS3, mais les ventes ont continué à fluctuer par la suite.

    3. **Impact des Consoles** :
        La sortie des PS2 et PS3 ont eu un impact direct sur les ventes, avec des titres majeurs comme *Final Fantasy X* et *Final Fantasy XIII*, contribuant à maintenir la franchise dans un marché compétitif.

    ##### Tomb Raider :
    1. **Pic Initial et Reboot** :
        Le succès initial de 1996 a été consolidé avec les suites, bien que les ventes aient progressivement diminué jusqu'au reboot de 2013, qui a relancé la franchise.

    2. **Comparaison avec Final Fantasy** :
        Comparé à *Final Fantasy*, *Tomb Raider* montre une stabilité relative dans ses ventes, avec des pics en 1997 et 2013. Final Fantasy, malgré des échecs, a su maintenir un volume de ventes plus élevé.

    #####  Duke Nukem :
    1. **Début Prometteur mais Déclin** :
        - *Duke Nukem* a connu ses meilleures ventes dans les années 90, mais le long développement de *Duke Nukem Forever* a coûté cher à la série.
        - Le retour en 2011 a été limité, et la série n'a pas retrouvé son ancienne gloire.

    ##### Comparaison globale entre les trois séries :
    1. **Final Fantasy** : Une domination marquée par l'innovation.
    2. **Tomb Raider** : Stabilité avec des pics occasionnels.
    3. **Duke Nukem** : Un potentiel non réalisé, limité par le manque de nouveautés.
    """)

    # Répartition des ventes globales par série et par année (DataFrame pour graphique empilé)
    df_pivot = df_cleaned.pivot_table(index='Year', columns='Series', values='Global_Sales', aggfunc='sum').fillna(0)
    fig = go.Figure()

    # Ajouter des barres empilées pour chaque série
    for series in df_pivot.columns:
        fig.add_trace(go.Bar(
            x=df_pivot.index,
            y=df_pivot[series],
            name=series,
            marker=dict(color={'Final Fantasy': 'blue', 'Tomb Raider': 'green', 'Duke Nukem': 'red'}[series])
        ))

    # Mise en page du graphique
    fig.update_layout(
        title='Ventes globales par série et par année (Barres empilées)',
        xaxis_title='Année',
        yaxis_title='Ventes globales (en millions)',
        barmode='stack',
        legend_title='Série',
        template='plotly_white',
        xaxis=dict(
            tickmode='linear',
            tick0=df_pivot.index.min(),
            dtick=1  # Afficher chaque année
        ),
        width=1200,  # Ajuster la largeur
        height=600   # Ajuster la hauteur
    )
    
    with st.container():
        st.plotly_chart(fig)

    if st.checkbox("Afficher le graphique des ventes par région"):
        fig_regional_sales = plot_all_series_regional_sales_plotly(df_cleaned, series_colors)
        with st.container():
            st.plotly_chart(fig_regional_sales)

    st.write("""
    Le graphique suivant montre la répartition des ventes globales par série de jeux vidéo (Final Fantasy, Tomb Raider, Duke Nukem) sur plusieurs années. Il permet de visualiser comment les ventes se sont réparties entre ces séries au fil du temps, en soulignant les périodes clés où chaque série a dominé le marché.

    ### Année 1997 : Une Année Historique
    L'année 1997 se distingue comme une période charnière, marquée par la sortie de deux jeux emblématiques : Final Fantasy VII et Tomb Raider. Final Fantasy VII, souvent considéré comme une révolution dans le genre RPG, a eu un impact colossal, consolidant la position de la série sur le marché. Simultanément, Tomb Raider a également connu un immense succès, rivalisant avec Final Fantasy sur le plan des ventes. C'est l'une des rares années où Tomb Raider a réellement concurrencé Final Fantasy en termes de popularité.

    ### Année 2013 : Le Renouveau de Tomb Raider
    L'année 2013 marque un autre moment important, notamment pour Tomb Raider, avec la sortie du reboot de la série. Ce renouveau a permis à Tomb Raider de revivre et de surpasser temporairement Final Fantasy en termes de ventes cette année-là. Cependant, malgré ce succès, Tomb Raider n'a pas réussi à maintenir une popularité constante au même niveau que Final Fantasy sur l'ensemble de la période analysée.

    ### Dominance de Final Fantasy
    En observant l'ensemble du graphique, on note une dominance claire de Final Fantasy sur la durée. Cette série a non seulement connu des pics de ventes élevés lors des sorties de titres majeurs, mais elle a également réussi à maintenir une présence continue et significative sur le marché des jeux vidéo. La sortie de Final Fantasy X sur PlayStation 2 et Final Fantasy XIII sur PlayStation 3 sont des exemples de la capacité de la série à marquer chaque génération de consoles.

    ### Duke Nukem : Une Présence Modeste
    En comparaison, Duke Nukem apparaît comme une série beaucoup plus modeste, loin de concurrencer les deux géants que sont Final Fantasy et Tomb Raider. Toutefois, le graphique montre un certain regain de ventes en 2011, probablement lié à un effet de nostalgie parmi les joueurs. Ce rebond, bien que limité, témoigne de l'impact initial que la série a pu avoir, malgré une incapacité à innover et à se maintenir au niveau des autres grandes franchises.
    """)

    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Les trois séries du premier graphique illustrent des trajectoires distinctes dans l'industrie du jeu vidéo. *Final Fantasy* incarne l'innovation et la croissance mondiale, malgré des fluctuations. *Tomb Raider* se distingue par sa stabilité et sa capacité à se réinventer, malgré des périodes de déclin. Quant à *Duke Nukem*, la série prometteuse n'a pas su évoluer avec le marché. *Final Fantasy* reste la plus influente des trois, mais chaque série a marqué l'industrie à sa manière.</span>""", unsafe_allow_html=True)

    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Le deuxième graphique montre la domination de *Final Fantasy* sur le marché, la résilience de *Tomb Raider* après des périodes difficiles, et les difficultés de *Duke Nukem* à rester pertinent. L'année 1997 a vu *Tomb Raider* rivaliser avec *Final Fantasy*, tandis que 2013 a marqué le renouveau de la série.</span>""", unsafe_allow_html=True)

    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">En comparant les régions, *Final Fantasy* domine au Japon, en Amérique du Nord et en Europe, consolidant son statut d'icône mondiale. *Tomb Raider* a un équilibre de ventes entre l'Amérique du Nord et l'Europe, mais peine au Japon. Enfin, *Duke Nukem* reste centré sur le marché nord-américain, avec peu de portée internationale. Ces différences soulignent l'importance d'une stratégie régionale bien adaptée aux marchés cibles.</span>""", unsafe_allow_html=True)

    st.subheader("Analyse Comparative des Ventes Globales")
    series_list = df_cleaned['Series'].unique()
    selected_series = st.selectbox("Choisissez une série", series_list)
    df_series = df_cleaned[df_cleaned['Series'] == selected_series]

    # Deuxième menu pour sélectionner le type d'analyse
    analysis_options = ['Plateformes', 'Éditeurs', 'Genre']
    selected_analysis = st.selectbox("Choisissez l'analyse", analysis_options)

    # 1. Analyse des ventes par plateforme
    if selected_analysis == 'Plateformes':
        sales_by_platform = df_series.groupby('Platform')['Global_Sales'].sum().reset_index()
    
        fig_platform = px.bar(
            sales_by_platform, 
            x='Platform', 
            y='Global_Sales', 
            color_discrete_sequence=[series_colors[selected_series]],
            title=f'Ventes Globales par Plateforme - {selected_series}'
        )
        fig_platform.update_layout(
            width=1000,  # Agrandir la largeur
            height=600,  # Agrandir la hauteur
            xaxis_title='Plateforme',
            yaxis_title='Ventes Globales (en millions)',
            title_x=0.5  # Centrer le titre
        )
        with st.container():
            st.plotly_chart(fig_platform)

    # 2. Analyse des ventes par éditeur
    elif selected_analysis == 'Éditeurs':
        sales_by_publisher = df_series.groupby('Publisher')['Global_Sales'].sum().reset_index()
    
        fig_publisher = px.bar(
            sales_by_publisher, 
            x='Publisher', 
            y='Global_Sales', 
            color_discrete_sequence=[series_colors[selected_series]],
            title=f'Ventes Globales par Éditeur - {selected_series}'
        )
        fig_publisher.update_layout(
            width=1000,
            height=600,
            xaxis_title='Éditeur',
            yaxis_title='Ventes Globales (en millions)',
            title_x=0.5
        )
        with st.container():
            st.plotly_chart(fig_publisher)

    # 3. Analyse des ventes par genre
    elif selected_analysis == 'Genre':
        sales_by_genre = df_series.groupby('Genre')['Global_Sales'].sum().reset_index()
    
        fig_genre = px.bar(
            sales_by_genre, 
            x='Genre', 
            y='Global_Sales', 
            color_discrete_sequence=[series_colors[selected_series]],
            title=f'Ventes Globales par Genre - {selected_series}'
        )
        fig_genre.update_layout(
            width=1000,
            height=600,
            xaxis_title='Genre',
            yaxis_title='Ventes Globales (en millions)',
            title_x=0.5
        )
        with st.container():
            st.plotly_chart(fig_genre)

    if selected_analysis == "platforme":
        st.write("""
        ##### Analyse Comparative des Ventes Globales par Plateforme : 
        Ce graphique présente une comparaison des ventes globales des trois franchises : **Final Fantasy**, **Tomb Raider**, et **Duke Nukem**, réparties par différentes plateformes. L'analyse met en évidence les performances relatives de ces séries en termes de ventes sur les principales consoles de jeu, tout en soulignant l'influence des plateformes sur le succès commercial de chaque franchise.
        """)

        # Create the Plotly figure
        fig = px.bar(
            df_cleaned, 
            x='Platform', 
            y='Global_Sales', 
            color='Series', 
            barmode='group',
            category_orders={'Platform': df_cleaned['Platform'].unique()},
            color_discrete_map={
                'Final Fantasy': 'blue',
                'Tomb Raider': 'green',
                'Duke Nukem': 'red'
            },
            labels={'Global_Sales': 'Ventes globales (en millions)', 'Platform': 'Plateforme'},
            title="Ventes moyennes globales par série et par plateforme"
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            title_x=0.5,
            legend_title_text='Séries',
            xaxis_title="Plateforme",
            yaxis_title="Ventes moyennes globales (en millions)"
        )
        
        st.plotly_chart(fig)

        st.write("""
        ##### Final Fantasy (Bleu)
        - **Performance Globale** : Final Fantasy domine clairement en termes de ventes globales sur plusieurs plateformes. Les consoles de Sony, notamment la PlayStation (PS), la PlayStation 2 (PS2), et la PlayStation 3 (PS3), ont été des piliers pour la série. Ces plateformes ont permis à la franchise de toucher un large public, consolidant sa position comme l'une des séries de jeux de rôle les plus vendues au monde.
        - **Influence des Plateformes** : La capacité de Final Fantasy à être présente sur presque toutes les plateformes majeures, y compris celles de Nintendo et Microsoft, a largement contribué à la longévité de la franchise. Cette stratégie d'accessibilité a permis à la série de rester pertinente à travers plusieurs générations de consoles et de capturer un public mondial varié.

        ##### Tomb Raider (Vert)
        - **Performance Globale** : Tomb Raider affiche une solide performance sur plusieurs plateformes, avec des ventes globales notables sur la PlayStation (PS), la PlayStation 3 (PS3), et la Xbox 360.
        - **Influence des Plateformes** : La polyvalence de Tomb Raider, qui a réussi à performer à la fois sur les plateformes Sony et Microsoft, montre que la série a su capter un public diversifié et s'adapter aux évolutions du marché.

        ##### Duke Nukem (Rouge)
        - **Performance Globale** : Duke Nukem montre des ventes globales plus modestes comparées aux autres franchises, ses meilleures performances étant sur des plateformes comme la Nintendo 64 (N64) et la Xbox 360.
        - **Influence des Plateformes** : Le succès limité de Duke Nukem sur une sélection restreinte de plateformes indique une difficulté à maintenir l'engagement des joueurs sur les nouvelles générations de consoles.

        Le graphique met en lumière l'importance des plateformes dans le succès commercial des franchises de jeux vidéo.
        """)
    elif selected_analysis == "Éditeurs":
        st.write("""
        #### Analyse des Ventes Globales par Éditeur pour les Séries

        Les éditeurs jouent un rôle déterminant dans le succès commercial des jeux vidéo, gérant à la fois la production et le marketing des jeux. Cette section examine les éditeurs qui ont eu le plus grand impact sur les ventes globales des franchises Final Fantasy, Tomb Raider, et Duke Nukem.

        ##### Final Fantasy (Blue)
        - **Éditeur Dominant** : SquareGroup (Square, Square Enix, Square EA, et SquareSoft) est l'éditeur principal de la série, responsable de plus de 80 millions de ventes globales. Sony Computer Entertainment a aussi joué un rôle crucial dans le succès des titres majeurs Final Fantasy VII et X.

        ##### Tomb Raider (Vert)
        - **Éditeurs Dominants** : Eidos Interactive a marqué les premiers succès de Tomb Raider avec plus de 30 millions de ventes globales, tandis que SquareGroup a relancé la série avec le reboot de 2013.

        ##### Duke Nukem (Rouge)
        - **Éditeur Principal** : Take-Two Interactive a publié les titres les plus vendus de la série Duke Nukem, bien que les ventes globales soient restées modestes par rapport aux autres franchises.
        """)

    # Sélection de la série et du type d'analyse
    st.subheader("Classement : Top et Flop par Série de Jeux")
    series_list = df_combined['Series'].unique()
    selected_series = st.selectbox("Choisissez une série", series_list, key="series_select")
    analysis_type = st.selectbox("Choisissez Top ou Flop", ['Top 5', 'Flop 5'], key="analysis_select")

    # Define the columns for plotting
    colG, colH = st.columns(2)
    df_series = df_combined[df_combined['Series'] == selected_series]

    # Définir les couleurs dégradées selon la série sélectionnée
    if 'Note' not in df_series.columns or 'Global_Sales' not in df_series.columns:
        st.error("Les colonnes 'Note' ou 'Global_Sales' ne sont pas présentes dans les données.")
    else:
        if selected_series == 'Final Fantasy':
            color_scale = 'Blues'
        elif selected_series == 'Tomb Raider':
            color_scale = 'Greens'
        elif selected_series == 'Duke Nukem':
            color_scale = 'Reds'
        else:
            color_scale = 'Greys'

    # Afficher le top 5 ou bottom 5 selon l'option choisie
    if analysis_type == 'Top 5':
        top_bottom_games = df_series.nlargest(5, 'Global_Sales')
        title = f'Top 5 jeux par ventes globales - {selected_series}'
    else:
        top_bottom_games = df_series.nsmallest(5, 'Global_Sales')
        title = f'Flop 5 jeux par ventes globales - {selected_series}'

    # Ajouter des commentaires analytiques selon la série et le type d'analyse
    if selected_series == 'Final Fantasy' and analysis_type == 'Top 5':
        st.write("""
        ##### Final Fantasy - Top 5
        1. **Final Fantasy VII (1997)** - Avec près de 10 millions de ventes, ce jeu a révolutionné le RPG.
        2. **Final Fantasy VIII (1999)** - Il a poursuivi le succès de la série avec 8 millions de ventes.
        3. **Final Fantasy X (2001)** - Une étape importante avec plus de 8 millions de ventes sur la PS2.
        """)
    elif selected_series == 'Final Fantasy' and analysis_type == 'Flop 5':
        st.write("""
        ##### Final Fantasy - Flop 5
        1. **Final Fantasy XI: All-in-One Pack 2006** - Les packs d'extensions ont généré peu de ventes comparé aux jeux principaux.
        2. **Dissidia: Final Fantasy Universal Tuning (2009)** - Ce jeu, bien qu'apprécié, a eu des ventes modestes en raison de son créneau de marché.
        """)
    elif selected_series == 'Tomb Raider' and analysis_type == 'Top 5':
        st.write("""
        ##### Tomb Raider - Top 5
        1. **Tomb Raider (1996)** - Avec plus de 5 millions de ventes, ce jeu a marqué l'industrie.
        2. **Tomb Raider II (1997)** - Un énorme succès avec plus de 7 millions de ventes, consolidant Lara Croft comme une icône.
        """)
    elif selected_series == 'Tomb Raider' and analysis_type == 'Flop 5':
        st.write("""
        ##### Tomb Raider - Flop 5
        1. **Tomb Raider: The Prophecy (2002)** - Ce titre sur Game Boy Advance a eu des ventes faibles en raison des limitations techniques.
        2. **Tomb Raider Chronicles (1999)** - N'a pas capté l'attention des joueurs en raison de la stagnation de la franchise à l'époque.
        """)
    elif selected_series == 'Duke Nukem' and analysis_type == 'Top 5':
        st.write("""
        ##### Duke Nukem - Top 5
        1. **Duke Nukem Forever (2011)** - Bien que critiqué, ce jeu a atteint 2 millions de ventes.
        2. **Duke Nukem: Time to Kill (1998)** - Ce jeu a su captiver les fans avec environ 1,5 million de ventes.
        """)
    elif selected_series == 'Duke Nukem' and analysis_type == 'Flop 5':
        st.write("""
        ##### Duke Nukem - Flop 5
        1. **Duke Nukem Trilogy: Critical Mass (2011)** - Un échec commercial avec des ventes très faibles.
        2. **Duke Nukem: Land of the Babes (2000)** - N'a pas su convaincre les joueurs, ce qui se reflète dans les ventes.
        """)

    with colG:
        fig = px.bar(top_bottom_games, x='Name', y='Global_Sales',
                      title=title, color='Global_Sales',
                      color_continuous_scale=color_scale,
                      labels={'Name': 'Jeu', 'Global_Sales': 'Ventes globales (en millions)'})

        # Center and enlarge the graph
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            title_x=0.5,  # Center the title
        )
        st.plotly_chart(fig)

    with colH:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis
        sns.scatterplot(data=top_bottom_games, x='Global_Sales', y='Note', hue='Name', palette='viridis', s=200, edgecolor='black', ax=ax)
        ax.set_title(f'Distribution des notes des {analysis_type.lower()} - {selected_series}')
        ax.set_xlabel('Ventes globales (en millions)')
        ax.set_ylabel('Note')
        ax.legend(title='Jeu', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        st.pyplot(fig)

        if selected_series == 'Final Fantasy':
            st.write("""Un constat important : le meilleur jeu en termes de ventes n'est pas nécessairement le mieux noté. Par exemple, bien que Final Fantasy VII domine en termes de ventes, c'est Final Fantasy VI qui obtient la meilleure note, tout en faisant partie des jeux les moins vendus.""")
        elif selected_series == 'Tomb Raider':
            st.write("""Pour Tomb Raider, les jeux avec des ventes élevées ont tendance à recevoir des notes globalement bonnes, à l'exception de certains cas isolés comme Tomb Raider: Underworld. D'autre part, certains jeux bien notés n'ont pas réussi à atteindre des ventes élevées, ce qui suggère que les ventes ne sont pas uniquement dictées par les critiques, mais également par d'autres facteurs, tels que la promotion et le timing de la sortie. Cela contraste avec la série Final Fantasy, où les notes varient beaucoup plus, et où il est moins fréquent de trouver une corrélation aussi directe entre la qualité perçue et le succès commercial.""")
        elif selected_series == 'Duke Nukem':
            st.write("""Le jeu Duke Nukem Trilogy: Critical Mass (2011) a été un échec commercial, avec des ventes très faibles. Le jeu a été mal accueilli en raison de son gameplay daté et de sa faible qualité de production, ce qui a largement contribué à son échec. Les jouers n'ont même pas noté le jeu.  
            Le faible nombre de jeux et l'insuffisance de données disponibles concernant la série Duke Nukem rendent difficile une analyse approfondie des facteurs influençant les ventes.""")

    st.subheader('Analyse des facteurs influençant les ventes des jeux')
    selected_series2 = st.selectbox("Choisissez une série", ['Final Fantasy', 'Tomb Raider'], key="series_select2")
    type_note_param1 = st.selectbox("Choisissez entre les paramètres suivants pour X-axis", ['Note', 'Etoiles', 'Envie de jouer'], key="note_select1")
    type_note_param2 = st.selectbox("Choisissez entre les paramètres suivants pour Y-axis", ['Coup de coeur', 'Etoiles', 'Envie de jouer'], key="note_select2")

    # Mapping for correct column names
    column_mapping = {
        'Note': 'Note',
        'Etoiles': 'Etoiles',
        'Envie de jouer': 'Envie_de_jouer',
        'Coup de coeur': 'Coup_de_coeur'
    }

    # Map selected parameters to DataFrame column names
    x_param = column_mapping[type_note_param1]
    y_param = column_mapping[type_note_param2]
    filtered_df = df_combined[df_combined['Series'] == selected_series2]

    # Create the scatter plot with selection-based axes
    fig = px.scatter(filtered_df,
                    x=x_param,
                    y=y_param,
                    color='Global_Sales',
                    size='Global_Sales',
                    hover_name='Name',
                    title=f'Relation entre {x_param} et {y_param}',
                    labels={x_param: x_param, y_param: y_param},
                    color_continuous_scale='Viridis')

    fig.update_layout(title_text=f'Relation entre {x_param} et {y_param}',
                    xaxis_title=x_param,
                    yaxis_title=y_param)

    st.plotly_chart(fig)

    if selected_series == 'Final Fantasy':
        st.write("""
        ##### Final Fantasy:
        Les ventes globales, symbolisées par la taille des cercles, varient indépendamment des notes, des étoiles, et même de l'envie de jouer, ce qui suggère que le succès commercial ne repose pas sur un seul facteur mais sur une multitude de variables interconnectées.""")
    elif selected_series == 'Tomb Raider':
        st.write("""
        ##### Tomb Raider:
        En conclusion, bien que les corrélations entre les différentes variables et le succès commercial des jeux Tomb Raider ne soient pas systématiques, certaines tendances se dégagent tout de même. Par exemple, les jeux ayant un grand nombre d’étoiles ou une forte envie de jouer tendent à recevoir un nombre plus important de coups de cœur et à générer des ventes plus élevées, mais cela reste limité et n’est pas applicable à tous les jeux de la série. Il est également important de noter que les ventes globales semblent influencées par des facteurs externes, comme la stratégie de marketing ou la notoriété de la franchise, qui ne sont pas entièrement capturés par les données disponibles. Cela montre que le succès commercial des jeux Tomb Raider dépend d'une combinaison de facteurs à la fois internes (évaluations des joueurs) et externes (promotion et contexte).""")
    
    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Pour mieux comprendre les raisons des ventes et des évaluations, nous allons analyser les critiques d’une sélection de jeux, en se basant sur les tops et flops. Quels sont les facteurs qui suscitent l'envie de jouer, ou au contraire, provoquent la déception ?</span>""", unsafe_allow_html=True)
    st.markdown("""
    Voici les jeux que nous avons choisis :
    1.	Final Fantasy VII (1997, 2006, 2007, 2020, 2022, 2023, 2024)
    2.	Final Fantasy VI (1994, 2006, 2014, 2022)
    3.	Tomb Raider II (1997, 1999)
    4.	Tomb Raider (2013)
    5.	Tomb Raider The prophecy (2002)
    6.	Duke Nukem (1997, 1998, 2011)
    """)

def display_sentiment():
    st.title("Analyse de Sentiments")
    st.write("""Nous avons utilisé VADER (Valence Aware Dictionary for Sentiment Reasoning) qui fournit des scores de sentiment en fonction des mots utilisés. C'est un
    analyseur de sentiment basé sur des règles, dans lequel les termes sont généralement étiquetés selon leur orientation sémantique comme étant soit positif, soit négatif
    ou neutre. Nous avons testé VADER de 2 bibliothèques différentes : nltk (Natural Language Toolkit) et vadersentiment.
    Tout d'abord, nous avons créé un analyseur d'intensité de sentiment pour catégoriser notre jeu de données. Ensuite, nous avons utilisé la méthode des scores de polarité 
    pour déterminer le sentiment qui sont mise dans différentes colonnes. Sur base de scores de polarité, une colonne ‘type’ a été ajouté pour indiquer le sentiment :
    **positif, négatif ou neutre.**""")

    st.subheader("Résultats de l'Analyse des Sentiments")
    data = {
        'Librairie': ['vadersentiment', 'nltk'],
        'Positifs': [418, 418],
        'Négatifs': [524, 522],
        'Neutres': [152, 154]
    }

    df = pd.DataFrame(data)
    st.table(df)

    dataset_choice = st.selectbox("Choisi une série :", 
                                   ["Final Fantasy", "Tomb Raider", "Duke Nukem"])

    if dataset_choice == "Final Fantasy":
        fig1 = create_histogram(df_FF_hist, "Sentiment Analysis sur la série Final Fantasy")
    elif dataset_choice == "Tomb Raider":
        fig1 = create_histogram(df_TR_hist, "Sentiment Analysis sur la série Tomb Raider")
    elif dataset_choice == "Duke Nukem":
        fig1 = create_histogram(df_DN_hist, "Sentiment Analysis sur la série Duke Nukem")
    
    st.plotly_chart(fig1)

    st.write("""Nous constatons **certaines limites** de l’analyse des sentiments. Il est particulièrement évident que les doubles négatifs ou les négations sont souvent mal interprétés, tout comme le langage sarcastique ou ironique. """)

    st.subheader("Nuages des mots")
    col1, col2 = st.columns([1, 1])
    choix_serie = col1.selectbox("Choisissez une série :", 
                                   ["Final Fantasy VI", "Final Fantasy VII", "Tomb Raider", "Duke Nukem"])
    
    mask_dict = {
        "Final Fantasy VI": "Logo FF6.png",
        "Final Fantasy VII": "Logo FF7.png",
        "Tomb Raider": "Logo TR.png",
        "Duke Nukem": "Logo DN.png"
    }

    mask_image = mask_dict[choix_serie]
    
    if choix_serie == "Final Fantasy VI":
        general_wordcloud_fig = plot_wordcloud(data_FF6['clean_lemmatized'], mask_image)
    elif choix_serie == "Final Fantasy VII":
        general_wordcloud_fig = plot_wordcloud(data_FF7['clean_lemmatized'], mask_image)
    elif choix_serie == "Tomb Raider":
        general_wordcloud_fig = plot_wordcloud(data_TR['clean_lemmatized'], mask_image)
    elif choix_serie == "Duke Nukem":
        general_wordcloud_fig = plot_wordcloud(data_DN['clean_lemmatized'], mask_image)
    
    col1.pyplot(general_wordcloud_fig)

    choix_sentiment = col2.selectbox("Choisissez un sentiment :", 
                                   ["POSITIF", "NEGATIF", "NEUTRAL"])
    
    if choix_serie == "Final Fantasy VI":
        filtered_data = data_FF6[data_FF6['type'] == choix_sentiment]
    elif choix_serie == "Final Fantasy VII":
        filtered_data = data_FF7[data_FF7['type'] == choix_sentiment]
    elif choix_serie == "Tomb Raider":
        filtered_data = data_TR[data_TR['type'] == choix_sentiment]
    elif choix_serie == "Duke Nukem":
        filtered_data = data_DN[data_DN['type'] == choix_sentiment]

    if not filtered_data.empty:
        sentiment_wordcloud_fig = plot_wordcloud(filtered_data['clean_lemmatized'], mask_image)  # Modify to appropriate mask image.
        col2.pyplot(sentiment_wordcloud_fig)
    else:
        st.write("Aucune donnée disponible pour le sentiment sélectionné.")

    st.markdown("""Les nuages de mots ainsi que les graphiques illustrent clairement que les joueurs accordent une grande importance aux personnages, au gameplay et
    à l'histoire des jeux. Les critiques, qu'elles soient positives ou négatives, utilisent souvent des termes similaires pour exprimer leurs opinions. Ainsi, ces éléments
    communs montrent comment les attentes des joueurs sont façonnées par leurs expériences antérieures et leur attachement à ces franchises.""")

    st.subheader("Analyse Sémantique : Mots Communs dans les Critiques des Joueurs") 
    top_n = st.slider("Nombre de mots à afficher :", 5, 25, 15)
    col3, col4 = st.columns([1, 1])

    if choix_serie == "Final Fantasy VI":
        plot_mots_pos = plot_most_frequent_words(data_FF6, "POSITIF", 'clean_lemmatized', 'Final Fantasy VI', top_n)
    elif choix_serie == "Final Fantasy VII":
        plot_mots_pos = plot_most_frequent_words(data_FF7, "POSITIF", 'clean_lemmatized', 'Final Fantasy VII', top_n)
    elif choix_serie == "Tomb Raider":
        plot_mots_pos = plot_most_frequent_words(data_TR, "POSITIF", 'clean_lemmatized', 'Tomb Raider', top_n)
    elif choix_serie == "Duke Nukem":
        plot_mots_pos = plot_most_frequent_words(data_DN, "POSITIF", 'clean_lemmatized', 'Duke Nukem', top_n)

    col3.pyplot(plot_mots_pos)

    if choix_serie == "Final Fantasy VI":
        plot_mots_neg = plot_most_frequent_words(data_FF6, "NEGATIF", 'clean_lemmatized', 'Final Fantasy VI', top_n)
    elif choix_serie == "Final Fantasy VII":
        plot_mots_neg = plot_most_frequent_words(data_FF7, "NEGATIF", 'clean_lemmatized', 'Final Fantasy VII', top_n)
    elif choix_serie == "Tomb Raider":
        plot_mots_neg = plot_most_frequent_words(data_TR, "NEGATIF", 'clean_lemmatized', 'Tomb Raider', top_n)
    elif choix_serie == "Duke Nukem":
        plot_mots_neg = plot_most_frequent_words(data_DN, "NEGATIF", 'clean_lemmatized', 'Duke Nukem', top_n)

    col4.pyplot(plot_mots_neg)

    st.markdown("""<span style="color: #F2AA84; font-weight: bold;">Dans les critiques de 'Final Fantasy', les mots clés qui ressortent incluent "combat", "musique" et "monde", soulignant l'importance des mécanismes
    de jeu et de l'ambiance. Pour 'Tomb Raider', des termes comme "Lara Croft" et "aventure" mettent en avant le personnage central et l’expérience immersive. Quant à
    'Duke Nukem', des mots tels que "humour", "fun" et "armes" reflètent le ton décalé et l'aspect ludique de la série.</span>""", unsafe_allow_html=True) 

def display_conclusion():
    controller_icon = "controller.png"
    st.title("Conclusion")
    st.markdown("""Ce projet a permis d'explorer en profondeur l'évolution et l'impact de trois franchises emblématiques du jeu vidéo : Final Fantasy, Tomb Raider et
    Duke Nukem. Chacune de ces séries a traversé des hauts et des bas, témoignant de l'évolution des préférences des joueurs et des dynamiques du marché.""") 

    col1, col2 = st.columns([0.02, 1])
    with col1:
        st.image(controller_icon, width=20)
    with col2:
        st.markdown("**Final Fantasy** se démarque par son succès mondial inégalé, avec une capacité à innover et à capturer l'intérêt des joueurs sur de multiples plateformes, bien que ses ventes aient commencé à décliner après 2005.")

    col1, col2 = st.columns([0.02, 1])
    with col1:
        st.image(controller_icon, width=20)
    with col2:
        st.markdown("**Tomb Raider**, avec son personnage iconique et sa nature d'action-aventure, a su s'adapter au fil des ans, se maintenant avec une base de fans fidèle, même si elle n'a pas réussi à s'imposer sur le marché asiatique.")

    col1, col2 = st.columns([0.02, 1])
    with col1:
        st.image(controller_icon, width=20)
    with col2:
        st.markdown("En revanche, **Duke Nukem**, bien que culte, a vu son influence diminuer, principalement en raison de son incapacité à innover et à répondre aux attentes d'un marché dynamique.")

    st.markdown("""
    Les analyses de données et les visualisations, telles que les nuages de mots, ont révélé que les joueurs attachent une grande importance aux 
    <span style="color: #F2AA84; font-weight: bold;">personnages, au gameplay et à l'histoire</span>, quelles que soient les critiques, soulignant l'impact de la 
    <span style="color: #F2AA84; font-weight: bold;">nostalgie</span> dans le cas des remakes. 
    La diversité des mots clés utilisés dans les critiques met en lumière l'expérience des joueurs et leurs attentes vis-à-vis des franchises.
    """, unsafe_allow_html=True)

    st.markdown("En somme, ce projet souligne l'importance de l'adaptation et de l'innovation dans le secteur du jeu vidéo, tout en mettant en exergue comment différents facteurs, y compris le marketing et la réputation des franchises, influencent le succès commercial.")

    st.markdown("<h5 style='text-align: center; color: white;'>Nuage des mots général de la série Tomb Raider</h5>", unsafe_allow_html=True)
    general_wordcloud_fig = plot_wordcloud(data_TR['clean_lemmatized'], "Logo TR.png")
    st.pyplot(general_wordcloud_fig)

    st.header("Perspectives")
    st.markdown("""**Diversifier les platformes:**""") 
    st.markdown("""À l'avenir, il est essentiel que des franchises emblématiques comme Final Fantasy et Tomb Raider s'étendent à de nouvelles plateformes, y compris 
    le cloud et les mobiles. Cela permettra d'atteindre une nouvelle génération de joueurs sans accès à des consoles traditionnelles, élargissant ainsi leur audience
    et ravivant l'intérêt pour ces séries classiques.""") 

    st.markdown("""**Prédire les ventes:**""")
    st.markdown("""De plus, les analyses appliquées à ces franchises pourraient servir à prédire les ventes d'autres jeux, en prenant en compte des éléments comme 
    les ventes antérieures et les critiques. En identifiant des tendances dans différents genres, des modèles prédictifs pourraient aider les développeurs à mieux 
    comprendre le marché et à ajuster leurs produits aux attentes des consommateurs, tout en optimisant les stratégies de marketing et de distribution pour garantir
    le succès commercial des nouveaux titres.""")

# Containers for the layout
header = st.container()

with header:
    st.image("bande.jpg", use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: white;'>From Pixels to Plot</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Data Storytelling à travers des séries</h2>", unsafe_allow_html=True)

    # Option menu
    selected = option_menu(
        None,
        ["Introduction", "Données", "Exploration & Visualisation", "Analyse de Sentiments", "Conclusion"],
        icons=["controller", "database", "bar-chart", "cloud", "flag"],
        key='menu',
        orientation="horizontal"
    )
    
    # Display the appropriate content based on the menu selection
    if selected == "Introduction":
        display_introduction()
    elif selected == "Données":
        display_data()
    elif selected == "Exploration & Visualisation":
        display_visualisation()
    elif selected == "Analyse de Sentiments":
        display_sentiment()
    elif selected == "Conclusion":
        display_conclusion()





