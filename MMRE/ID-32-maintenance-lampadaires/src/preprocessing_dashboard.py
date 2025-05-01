################################################################################
# Script Name  : preprocessing_dashboard.py
# Description  : Préparation des données pour le dashboard
# Auteur       : basile@mitsiomotu.com
# Date : 2025/05/01
################################################################################

from preprocessing_lampadaires import preprocess_lampadaires_prise
from lampadaires_grappes_dysfonctionnement import find_grappes
from lampadaires_analyse_textuelle import main

def read_data():

    df_prise = preprocess_lampadaires_prise()

    df_grappes = find_grappes()[0]

    return df_prise, df_grappes


def flag_grappes(df_prise, df_grappes):
    # id des lampadaires dans les grappes
    id_grappes = df_grappes["id"].copy()

    # flag si l'id est dans les grappes
    df_prise["est_dans_grappe"] = df_prise["id"].isin(id_grappes).astype(int)

    return df_prise

def clean_data(df):

    df = map_region(df)

    df = remplacer_ampoules(df)

    df =remplacer_type(df)

    # on considère que les non renseignés sont fonctionnels
    # H0 prudente cf eda
    df["est_fonctionnel"] = df["est_fonctionnel"].fillna(1)

    return df.drop(columns=["source"])


def map_region(df):
    # Assigne 'Grand Lomé' si la préfecture est Golfe ou Agoè-Nyivé
    df.loc[df["prefecture"].isin(["Golfe", "Agoè-Nyivé"]), "region"] = "Grand Lomé"

    return df


def remplacer_ampoules(df):
    # Dictionnaire de mapping
    ampoules_mapping = {
    "LED": "LED",
    "HPS/LPS": "Sodium"
    }

    df["ampoules_type"] = df["ampoules_type"].map(ampoules_mapping).fillna("Autres")
    return df


def remplacer_type(df):
    # Dictionnaire de mapping
    type_mapping = {
        "Nsp": "Réseaux",
        "Informel" : "Réseaux"
    }

    df["type"] = df["type"].replace(type_mapping)
    return df

def preprocess_for_dashboard():

    df_prise, df_grappes = read_data()

    df_prise = flag_grappes(df_prise, df_grappes)

    df_prise = clean_data(df_prise)

    return df_prise

def save_files(df):

    # pour mapbox
    df[(df["est_fonctionnel"] == 0) &
       (df["est_dans_grappe"] == 0)]\
       .to_file("./data/dashboard/maintenance_lampadaires.geojson")
    
    df["latitude"] = df.geometry.y
    df["longitude"] = df.geometry.x

    df.drop(columns="geometry")\
        .to_csv("./data/dashboard/maintenance_lampadaires.csv",
                sep=";",
                decimal = ",",
                index=False)

if __name__ == "__main__":

    df = preprocess_for_dashboard()

    save_files(df)

