################################################################################
# Script Name  : lampadaires_analyse_comment uelle.py
# Description  : Détermination des causes de dysfonctionnement par T.A.L
# Auteur       : basiledesj@hotmail.fr
# Date : 2025/04/22
################################################################################
"""
On a pu isoler les grappes de dysfonctionnement dû à des problèmes d'accès
à l'électricité (pb de boîtier, cables).
Pour les autres lampadaires dysfonctionnels, on recherche les causes de dysfonc
ctionnement en analysant les remarques notées dans le cadre de PRISE.
On distigue l'analyse des lampadaires solaires et réseaux
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy
import re
from wordcloud import WordCloud
from collections import Counter
import unidecode 

from preprocessing_lampadaires import preprocess_lampadaires_prise
from lampadaires_grappes_dysfonctionnement import find_grappes



def get_comments(df_prise):
    """
    Concatène les différents champs de commentaires
    et ne garde que champs non vides et informatifs
    """
    comments_1 = df_prise["pb"].replace(["Néant","Autre"], "").fillna("")
    comments_2 = df_prise["probleme_autre"].replace("Néant", "").fillna("")
    comments_3 = df_prise["probleme_cable_autre"].replace("Néant", "").fillna("")

    comments = comments_1 + "" + comments_2 + "" + comments_3

    useless_comments = [
        r"La lampe ne s'allume pas", 
        r"Lampe ne s'allume plus", 
        r"Lampe ne s'allume", 
        r"Lampe non allumée",
        r"La lampe ne fonctionne pas",
        r"Lampadaire plus fonctionnelle",
        r"Lampe non fonctionnelle",
        r"Le lampadaire ne s'allume pas",
        r"Poteaux penche",
        r"Poteau en bois non conventionnel",
        r"Poteau au sol",
        r"Poteau presque au sol",
        r"Niveau Poteau",
        r"Niveau poteau",
        r"Sous les arbres",
        r"Autre",
        r"Problème de batterie ou de lampe ou de câbles ou du panneau solaire",
        r"Problème de lampe ou de câbles",
        r"Problème de lampe ou câble",
        r"Ne s'allume pas",
        r"Ne s allume pas",
        r"Lampe mal positionnee",
        r"Lampe mal positionnée",
        r"Pas de problème",
        r"Éteint"
    ]
    
    comments = comments.replace(regex=useless_comments, value="")

    comments = comments.apply(remove_punctuation_and_spaces)
    return comments


def remove_punctuation_and_spaces(comment):

    # Supprimer la ponctuation
    comment = re.sub(r"[^\w\s]", " ", comment)

    # supprime les espaces en début et fin de groupe sde mots 
   # Remplacer les groupes multiples d'espaces ou virgules par un seul espace
    comment = re.sub(r"\s+", " ", comment)  # condense les espaces multiples
    comment = re.sub(r"(^\s+|\s+$)", "", comment)  # supprime début/fin

    return comment

def get_and_filter_data():

    df_prise = preprocess_lampadaires_prise(keep_comments=True)
    df_grappe, _ = find_grappes()

    # On ne garde que les lampadaires dysfonctionnels
    df_prise = df_prise[df_prise["est_fonctionnel"] == 0].copy()
    id_grappes = df_grappe.id.to_numpy()
    df_prise["in_grappe"] = np.isin(df_prise["id"], id_grappes)

    # Qui ont été commentés
    df_prise["comments"] = get_comments(df_prise)
    df_prise["is_commented"] = df_prise["comments"] != ""

    print("Analyse des lampadaires dysfonctionnels, commentés et hors grappe")
    print(df_prise.groupby(["in_grappe", "is_commented"]).id.count())

    # On analyse les lamapdaires hors grappes commentés
    df_study = df_prise[(df_prise["in_grappe"] == False)
                        &
                        (df_prise["is_commented"] == True)
                        ]


    df_study = df_study[["id", "comments", "type"]]


    print("Tous les types sont représentés\n", df_study.groupby("type").id.count())

    return df_study


def categorize_comment(comment):

    categories = {'capot': "capot",
     'ampoule' : ["lampe manquante","ampoule", "lampe cassee", "lampe gril"],
     'boitier' : ["boitier", "boîtier"],
     "batterie" : "batterie",
     "panneau" : "panneau",
     "cable" : "cables"
     }

    comment = comment.lower()

    matched = []
    for key, keywords in categories.items():
        if isinstance(keywords, str):
            keywords = [keywords]
        if any(kw in comment for kw in keywords):
            matched.append(key)

    if matched == []:
        return np.nan

    return matched

def categorize_comments(df_study):
    """
    Applique categorize_issues à chaque commentaire
    et crée une colonne booléenne pour chaque type de problème détecté
    """
    # Appliquer la catégorisation
    df_study["cat"] = df_study["comments"].apply(categorize_comment)

    # Initialiser les colonnes par catégorie
    all_categories = ['capot', 'ampoule', 'boitier', 'batterie', 'panneau', 'cable']
    for cat in all_categories:
        df_study[cat] = df_study["cat"].apply(lambda lst: cat in lst if isinstance(lst, list) else False)

    return df_study


def get_nlp_tools():
    # Télécharger les stopwords si ce n'est pas déjà fait
    nltk.download("stopwords")

    # Charger spacy (modèle français)
    nlp = spacy.load("fr_core_news_sm")
    lemmatizer = nlp

    # Stopwords FR
    stop_words = set(stopwords.words("french"))
    # Liste des mots personnalisés à ajouter
    #stopwords_personnalises = ["Niveau", "probleme", "autre", "lampadaire"]
    stopwords_personnalises = []

    # Ajouter les mots personnalisés
    stop_words.update(stopwords_personnalises)

    return lemmatizer,  stop_words

def preprocess_comment(comment, lemmatizer, stop_words):

    # Enlever accents + passer en minuscule
    comment = unidecode.unidecode(comment.lower())

    # Supprimer la ponctuation
    comment = re.sub(r"[^\w\s]", " ", comment)

    # Lemmatiser avec spaCy
    doc = lemmatizer(comment)
    words = [token.lemma_ for token in doc]

    # Enlever les stopwords et mots trop courts
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return words

def get_and_plot_wordcloud(comments):

    lemmatizer, stop_words = get_nlp_tools()

    all_words = []
    for comment in comments:
        all_words.extend(preprocess_comment(comment, lemmatizer, stop_words))

    # Fréquence des mots
    word_freq = Counter(all_words)

    # Générer le WordCloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate_from_frequencies(word_freq)

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Mots les plus fréquents dans les rapports de dysfonctionnement", fontsize=14)
    plt.show()


def main():

    df_study = get_and_filter_data()

    df_study = categorize_comment(df_study)
    
    return df_study

if __name__ == "__main__":

    #df_study = get_and_filter_data()

    #df_study = categorize_comments(df_study)
    
    df_study = main()

    # enleve ceux qui n'ont pas de catégorie (150)

    df_study = df_study[~df_study["cat"].isna()]

    # Liste des catégories utilisées
    issue_cols = ["capot", "ampoule", "boitier", "batterie", "panneau", "cable"]
    
    # farde que solaire et réseaux
    df_study = df_study[(df_study.type == "Solaire") | (df_study.type == "Réseaux")]

    # Total de lampadaires par type
    df_total = df_study.groupby("type").size().rename("total")

    # Nombre de lampadaires avec chaque problème, par type
    df_counts = df_study.groupby("type")[issue_cols].sum()

    # Pourcentage par rapport au total
    df_percent = df_counts.div(df_total, axis=0).add_suffix("_pct")

    # Tout combiner
    df_summary = pd.concat([df_total, df_percent], axis=1)

    # Afficher le tableau final
    print(df_summary)

