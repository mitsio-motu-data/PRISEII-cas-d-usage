################################################################################
# Script Name  : preprocessing_dashboard.py
# Description  : Processing des données pour le dashboard
# Auteur       : basile@mitsiomotu.com
# Date         : 2025/05/01
################################################################################

import pandas as pd
import geopandas as gpd
import numpy as np

def charger_donnees():
    df = pd.read_csv('./data/spatialisation_lits.csv')
    gdf = gpd.read_file("./data/15_01_25_Togo_Prefecture_4326.gpkg")
    return df, gdf

def ajouter_colonnes_base(df):
    df["nb_lits_maternite"] = df["lits_maternite_nbr_Prive"] + df["lits_maternite_nbr_Public"]
    df["nb_lits_hopital"] = df["lits_hopital_total_nbr_Prive"] + df["lits_hopital_total_nbr_Public"]
    df["nb_lits_hopital_pour_10_000_hab"] = 10_000 * df["nb_lits_hopital"] / df["Population"]
    df["nb_lits_maternite_pour_10_000_hab"] = 10_000 * df["nb_lits_maternite"] / df["Population"]
    return df



def sauvegarder(df, slider):
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("./data/repartition_spatiale_des_lits.geojson")
    df.drop(columns="geometry").to_csv("./data/repartition_spatiale_des_lits.csv", index=False, sep=";", decimal=",")
    slider.to_csv('./data/slider.csv', index=False, sep=";", decimal=",")

def main():
    df, gdf = charger_donnees()
    
    df = df.set_index("prefecture").join(
        gdf[["prefecture_nom", "geometry"]].set_index("prefecture_nom")
    ).reset_index()

    df["prefecture_id"] = df.index
    df = ajouter_colonnes_base(df)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf.crs)

    sauvegarder(df)

if __name__ == "__main__":
    main()
