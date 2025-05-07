################################################################################
# Script Name  : preprocessing_dashboard.py
# Description  : Preprocessing pour le dashboard
# Auteur       : basile@mitsiomotu.com
# Date : 2025/05/05
################################################################################

import pandas as pd
import geopandas as gpd
import os
import subprocess

def read_files():
    df_results = pd.read_csv('./data/results.csv')
    df_aires_urbaines = gpd.read_file('./data/main.gpkg', layer="aires_urbaines")\
                        [['cluster',
                        'region_nom',
                        'commune_nom',
                        'commune',
                        'geometry']]

    return df_results, df_aires_urbaines

def filter_clusters(df_results, df_aires_urbaines):
    clusters_results = df_results["Aire urbaine"]
    clusters_aires = df_aires_urbaines["cluster"]
    clusters_missing = list(set(clusters_aires) - set(clusters_results))

    return df_aires_urbaines[~df_aires_urbaines["cluster"].isin(clusters_missing)]

def transform_to_long_format(df_results):
    """
    Transforme df_results en format long avec les colonnes :
    - Aire urbaine, Region, Prefecture, Commune
    - type (primaire, secondaire, tertiaire)
    - lineaire_km, nb_lampadaire, nb_lampadaire_par_km
    """
    types = ['Primaire', 'Secondaire', 'Tertiaire']
    df_long = pd.DataFrame()

    for t in types:
        df_temp = df_results[[
            'Aire urbaine', 'Region', 'Prefecture', 'Commune',
            f'{t} - KM',
            f'{t} - Lampadaire',
            f'{t} lampadaires / km'
        ]].copy()

        df_temp.columns = [
            'Aire urbaine', 'Region', 'Prefecture', 'Commune',
            'lineaire_km', 'nb_lampadaire', 'nb_lampadaire_par_km'
        ]
        df_temp['type'] = t.lower()
        df_long = pd.concat([df_long, df_temp], ignore_index=True)

    return df_long[[
        'Aire urbaine', 'Region', 'Prefecture', 'Commune',
        'type', 'lineaire_km', 'nb_lampadaire', 'nb_lampadaire_par_km'
    ]]

def export_all(df_results, df_aires_urbaines):

    df_results.to_csv("./data/eclairage_aires_urbaines_bi.csv", 
                      index=False, sep=";", decimal=",")
    
    df_aires_urbaines\
        .to_crs("EPSG:4326")\
        .to_file("./data/eclairage_aires_urbaines_bi.geojson")

    #subprocess.run(
    #    "tippecanoe -o ./data/elcaira.mbtiles "
    #    "--force --drop-densest-as-needed -z 14 -Z 0 ./data/tinga.geojson",
    #    shell=True
    #)
    #os.remove("./data/tinga.geojson")


def main():
    df_results, df_aires_urbaines = read_files()
    df_aires_urbaines = filter_clusters(df_results, df_aires_urbaines)

    df_results = transform_to_long_format(df_results)

    export_all(df_results, df_aires_urbaines)

if __name__ == "__main__":
    main()