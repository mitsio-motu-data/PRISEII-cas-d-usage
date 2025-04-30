################################################################################
# Script Name  : preprocessing.py
# Description  : Preprocessinf pour le dashboard
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/29
################################################################################

import geopandas as gpd
import pandas as pd
from shapely.wkt import loads


def read_equipements():

    df_equip = pd.read_csv("./data/raw/eq_combined_restricted.csv", 
                    usecols=[
                        "id",
                        "geometry", 
                        "renaming", 
                        "secteur_grouped", 
                        "etab_name", 
                        "etab_id",
                        "region_nom_bdd",
                        "prefecture_nom_bdd" 

                    ])

    df_equip = df_equip.dropna(subset=["renaming"])

    df_equip["geometry"] = df_equip["geometry"].apply(loads)

    df_equip = gpd.GeoDataFrame(df_equip, geometry="geometry", crs="EPSG:4326")

    df_equip.to_crs(crs="EPSG:25231")

    df_equip = map_region(df_equip)
    return df_equip

def read_resultats():
    
    df_results = pd.read_csv("./data/raw/resultats.csv")

    return df_results

def join(df_etab, df_pref):
    return

def map_region(df):
    # Assigne 'Grand Lomé' si la préfecture est Golfe ou Agoè-Nyivé
    df.loc[df["prefecture_nom_bdd"].isin(["Golfe", "Agoè-Nyivé"]), "region_nom_bdd"] = "Grand Lomé"

    # Dictionnaire de correspondance des noms de régions vers des IDs
    region_to_id = {
        "Grand Lomé": 1,
        "Maritime": 2,
        "Plateaux": 3,
        "Centrale": 4,
        "Kara": 5,
        "Savanes": 6,
    }

    # Création de la colonne region_id
    df["region_id"] = df["region_nom_bdd"].map(region_to_id)

    return df

if __name__ == "__main__":

    df_equip = read_equipements()
    df_results = read_equipements()

    df_results.to_csv("./data/resultats.csv", sep=";", decimal=",", index=False)
    df_equip.to_file("./data/equipements.geojson", index=False)
    
    df_dashboard = df_equip.copy()
    
    df_dashboard["latitude"] = df_dashboard.geometry.y
    df_dashboard["longitude"] = df_dashboard.geometry.x

    df_dashboard = df_dashboard.drop(columns=["geometry"])

    df_dashboard.to_csv('./data/equipements.csv',
                        index=False,
                        sep=";", 
                        decimal=",")