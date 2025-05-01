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
    
    df_results = pd.read_csv("./data/raw/resultats.csv", sep=";")


    return df_results

def read_distance():

    df_distance = pd.read_csv('./data/raw/Indicateur_distance.csv', sep=";")

    df_distance = transformer_table_distance(df_distance)

    return df_distance


def transformer_table_distance(df):
    """
    Transforme une table de type wide en une table longue avec les colonnes :
    'région', 'équipement', 'valeur'.
    
    Paramètres :
        df (pd.DataFrame) : table d'entrée avec une colonne 'region_nom' et des colonnes d'équipements.
        
    Retour :
        pd.DataFrame : table transformée avec les colonnes 'région', 'équipement', 'valeur'.
    """
    df_long = df.melt(id_vars='region_nom', var_name='équipement', value_name='valeur')
    return df_long


def map_region(df):
    # Assigne 'Grand Lomé' si la préfecture est Golfe ou Agoè-Nyivé
    df.loc[df["prefecture_nom_bdd"].isin(["Golfe", "Agoè-Nyivé"]), "region_nom_bdd"] = "Grand Lomé"

    return df

if __name__ == "__main__":

    df_equip = read_equipements()
    df_results = read_resultats()
    df_distance = read_distance()

    df_results.to_csv("./data/resultats.csv",
                      sep=";",
                      decimal=",",
                      index=False)
    
    # pour mapbox
    df_equip.to_file("./data/equipements.geojson",
                     index=False)
    
    # pour power bi
    df_dashboard = df_equip.copy()
    
    df_dashboard["latitude"] = df_dashboard.geometry.y
    df_dashboard["longitude"] = df_dashboard.geometry.x

    df_dashboard = df_dashboard.drop(columns=["geometry"])

    df_dashboard.to_csv('./data/equipements.csv',
                        index=False,
                        sep=";", 
                        decimal=",")
    

    df_distance.to_csv("./data/distances.csv",
                       sep=";",
                       decimal=",",
                       index=False)