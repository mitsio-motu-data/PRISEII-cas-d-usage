################################################################################
# Script Name  : preprocessing.py
# Description  : Retravail fichier tinga pour power bi
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/28
################################################################################

import geopandas as gpd
import pandas as pd


def map_raccordables_to_agence(df_raccordables, df_agence):
    """
    Associe une agence à chaque zone raccordable, arbitrairement on choisit en 
    cas de chevauchement 
    """

    join = gpd.sjoin_nearest(df_raccordables, df_agence[["agence_id", "geometry"]], how="left")


    df_raccordables = join.drop_duplicates(subset="geometry")

    return df_raccordables.drop(columns="index_right")


def preprocess_raccordables(df_agence):
    """
    Dans le fichier original il n'y a qu'une seule géométrie pour toutes les zones
    raccordables, on la splitte en lignes simples
    """
    df_raccordable = gpd.read_file("./data/zone_1_raccordable.gpkg")[[
        "geometry"
    ]]

    multi = df_raccordable.geometry[0]

    simple_polygons = list(multi.geoms)

    gdf_polygons = gpd.GeoDataFrame(geometry=simple_polygons, crs=df_raccordable.crs)

    df_raccordable = map_raccordables_to_agence(gdf_polygons, df_agence)

    return df_raccordable

def preprocess_agences():

    df_agence = gpd.read_file("./data/agence_resultats_joint.gpkg")
    df_agence["agence_id"] = df_agence.index

    return df_agence

def preprocess_hexa(df_agence):
    """
    Assigne une agence à quasi chaque hexagone
    """
    
    df_hexa = gpd.read_file("./data/hexagones_classifies.gpkg")[[
        "hex_id",
        "classification",
        "pop_rgph_point_2024",
        #"region_nom",
        "geometry"
    ]]
    # simpler 

    join = gpd.sjoin(df_hexa, df_agence[["agence_id", "geometry"]])
    
    # 1% des hex sur les bords
    join = join.dropna(subset=["agence_id"]).drop(columns="index_right")
    
    return join

def read_data():    

    df_agence = preprocess_agences()

    df_raccordable = preprocess_raccordables(df_agence)

    df_hexa = preprocess_hexa(df_agence)

    return df_hexa, df_raccordable, df_agence


def dissolve_hexagones(df_hexa):
    """
    Rassembler les hexagone similaires de la même agence
    """

    # petit buffer
    df_hexa["geometry"] = df_hexa["geometry"].buffer(1)

    df_hexa_dissolved = df_hexa.dissolve(
            by=["classification", "agence_id"],
            aggfunc={
                #"region_nom": "first",
                "pop_rgph_point_2024": "sum",
            }
        ).reset_index()


    return df_hexa_dissolved

def reclassify(df_hexa_dissolved, df_raccordable):
    """
    Paver le plan avec zone raccordables + classification des hexagones
    """

    # hexagones qui ne sont pas raccordables
    df_hexa_restants = gpd.overlay(df_hexa_dissolved, df_raccordable, how="difference")

    df_raccordable["classification"] = "raccordable"

    # joindre les deux
    df_classif = pd.concat([df_raccordable, 
                            df_hexa_restants], ignore_index=True)

    mapping_new_cat = {'raccordable': 'Raccordable (< 60 m)',
                        'densification_rurale': 'Densification (< 1 km)',
                        'densification_urbaine': 'Densification (< 1 km)',
                        'extension_rurale': 'Extension',
                        'extension_urbaine': 'Densification ',
                        'zone_eloignee': 'Solution off-grid'}

    df_classif["id"] = df_classif.index
    return df_classif


def main():

    df_hexa, df_raccordable, df_agence = read_data()

    df_hexa_dissolved = dissolve_hexagones(df_hexa)
    df_classif = reclassify(df_hexa_dissolved, df_raccordable)

    return df_classif, df_agence


if __name__ == "__main__":

    df_classif, df_agence = main()

    df_classif.to_crs("EPSG:4326").to_file("./data/final.geojson")

    df_classif.drop(columns="geometry").to_csv('./data/tinga.csv', index=False)

    df_agence.drop(columns=["geometry"])\
        .fillna(0).to_csv("./data/resultats_par_agence.csv",
                          index=False,
                          decimal=",",
                          sep=";")

