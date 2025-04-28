################################################################################
# Script Name  : preprocessing.py
# Description  : Retravail fichier tinga pour power bi
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/28
################################################################################

import geopandas as gpd
import pandas as pd

def preprocess_raccordables():
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

    return gdf_polygons

def preprocess_agences():
    """
    Élargit uniquement les bords extérieurs des agences pour mieux couvrir la frontière du pays,
    sans modifier les limites internes entre agences.
    """
    df_agence = gpd.read_file("./data/agence_resultats_joint.gpkg")
    df_agence["agence_id"] = df_agence.index

    # 1. Calculer l'union de toutes les agences (la forme totale)
    union_geom = df_agence.unary_union

    # 2. Buffer sur l'union (ex : +50m pour élargir)
    union_buffered = union_geom.buffer(3_000)

    # 3. Créer une couche de "complément" entre ancien contour et élargi
    extension_geom = union_buffered.difference(union_geom)

    # buffer sur les agences 
    df_agence["geometry_buffer"] = df_agence["geometry"].buffer(3_000)

    # prend l'intersection entre geometry buffer et l'union


    # ajotue l'intersection aux agences initiales
   
    return df_agence

def preprocess_hexa(df_agence):
    
    df_hexa = gpd.read_file("./data/hexagones_classifies.gpkg")[[
        "hex_id",
        "classification",
        "commune_nom", 
        "region_nom",
        "prefecture_nom",
        "canton_nom",
        "pop_rgph_point_2024", 
        "nombre_de_batiments",
        "surface_batie_totale",
        "geometry"
    ]]
    df_hexa = merge_agences(df_hexa, df_agence)
    
    return df_hexa

def read_data():    

    df_agence = preprocess_agences()

    df_raccordable = preprocess_raccordables()

    df_hexa = preprocess_hexa(df_agence)

    return df_hexa, df_raccordable, df_agence


def dissolve_hexagones(df_hexa):
    """
    Rassembler les hexagone similaires de la même commune et agence
    """

    # petit buffer
    df_hexa["geometry"] = df_hexa["geometry"].buffer(1)

    df_hexa_dissolved = df_hexa.dissolve(
            by=["classification", "agence_id", "commune_nom"],
            aggfunc={
                "region_nom": "first",
                "prefecture_nom" :"first",
                "canton_nom": "first",
                "pop_rgph_point_2024": "sum",
                "nombre_de_batiments": "sum",
                "surface_batie_totale": "sum",
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

    #mapping_new_cat = {'raccordable': 'raccordable',
    #                    'densification_rurale': 'extension',
    #                    'densification_urbaine': '',
    #                    'extension_rurale': '',
    #                    'extension_urbaine': '',
    #                    'zone_eloignee': ''}

    df_classif["id"] = df_classif.index
    return df_classif




def main():

    df_hexa, df_raccordable, df_agence = read_data()

    df_hexa_dissolved = dissolve_hexagones(df_hexa)
    df_classif = reclassify(df_hexa_dissolved, df_raccordable)
    #df_classif = merge_agences(df_classif, df_agence)
    return df_classif


if __name__ == "__main__":

    df_classif = main()

