##########################################################################
# Script Name  : preprocessing_lampadaires.py
# Description  : Traite et harmonise les données de lampadaires issues du projet
# PRISE et des données SUNNA.
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/07
##########################################################################

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

PATH_PRISE = "./data/ener_lampadaires.gpkg"
PATH_SUNNA = "./data/2025-04-07_Extract_Base 50 000_MM pour QGIS.csv"

def clean_fonctionnel(df):
    """
    Le dataset initial contient trois colonnes, que l'on synthètise ici en une seule.

    Retourne:
        - pd.Series "est_fonctionnel" :
            - 0 : non
            - np.nan : nsp
            - 1 : oui
    """

    df["est_fonctionnel"] = df["fonctionnel"]

    return df.drop(columns=["non_fonctionenel", "fonctionnel", "fonction_nsp"])


def rename_columns_prise(df):
    column_mapping = {
        "id": "id",
        "source": "source",
        "region": "region",
        "prefecture": "prefecture",
        "canton": "canton",
        "commune": "commune",
        "batiment_nom": "batiment",
        "lampadaire_type": "type",
        "est_fonctionnel": "est_fonctionnel",
        "annee" : "annee",
        "geometry": "geometry",
    }

    # Renommer les colonnes
    df = df.rename(columns=column_mapping)


    return df


def preprocess_lampadaires_prise(path=PATH_PRISE, keep_comments=False):
    """
    Traite le fichier de lampadaires PRISE.
    Garde éventuellement les commentaires dans le fichier de lampadaires PRISE.
    """

    # colonnes à lire
    usecols = [
            "region",
            "prefecture",
            "commune",
            "canton",
            "id",
            "lampadaire_type",
            "batiment_nom",
            "fonctionnel",
            "non_fonctionenel",
            "fonction_nsp",
            "ampoules_type",
            "annee",
            "geometry",
        ]
    
    if keep_comments:
        usecols += ["pb", "probleme_autre", "probleme_cable_autre"]
    
    df = gpd.read_file(
        path,
        columns=usecols,
    )

    df = clean_fonctionnel(df)

    df["source"] = "PRISE"

    df = rename_columns_prise(df)

    return df


def rename_columns_sunna(df):

    column_mapping = {
        "ID du lamp.": "id",
        "source": "source",
        "Région": "region",
        "Préfecture": "prefecture",
        "Canton": "canton",
        "Commune": "commune",
        "Infrastructure": "batiment",
        "Type": "type",
        "est_fonctionnel": "est_fonctionnel",
        "geometry": "geometry",
    }

    # Renommer uniquement les colonnes mappées
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Réordonner les colonnes selon l'ordre du mapping
    desired_order = list(column_mapping.values())
    df = df[desired_order]

    return df


def preprocess_lampadaires_sunna(path=PATH_SUNNA):

    df = pd.read_csv(
        path,
        usecols=[
            "Région",
            "Préfecture",
            "ID du lamp.",
            "Commune",
            "Canton",
            "Infrastructure",
            "Latitude M",
            "Longitude M",
        ],
    )

    # Créer la géométrie et convertit en gdf
    geometry = [Point(xy) for xy in zip(df["Longitude M"], df["Latitude M"])]
    df["geometry"] = geometry
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["Latitude M", "Longitude M"]),
        geometry=geometry,
        crs="EPSG:4326",
    )

    # ajout des colonnes manquantes
    df["source"] = "SUNNA"
    df["est_fonctionnel"] = 1
    df["type"] = "Solaire"

    gdf = rename_columns_sunna(df)

    return gdf

def uniformize_prefecture_names(df):
    """erreurs dans le fichier sunna"""
    df["prefecture"] = df["prefecture"].replace(
        {
            "Akebou": "Akébou",
            "HAHO": "Haho",
        })
    
    return df


def preprocess_lampadaires():

    df_prise = preprocess_lampadaires_prise(PATH_PRISE)
    df_sunna = preprocess_lampadaires_sunna(PATH_SUNNA)

    crs = df_prise.crs

    df = pd.concat([df_prise, df_sunna], ignore_index=True)

    # Harmonisation des noms de préfectures
    df = uniformize_prefecture_names(df)
    
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    return df

if __name__ == "__main__":
    preprocess_lampadaires()