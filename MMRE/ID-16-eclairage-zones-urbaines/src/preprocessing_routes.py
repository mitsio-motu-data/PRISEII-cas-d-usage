################################################################################
# Script Name  : preprocessing_routes.py
# Description  : Recatégorise les routes Open Street Map. Le fichier source peut
# retrouvé à l'adresse suivante :
# https://s3.dualstack.us-east-1.amazonaws.com/production-raw-data-api/ISO3/TGO/
# roads/lines/hotosm_tgo_roads_lines_gpkg.zip
# (site source : https://data.humdata.org/dataset/hotosm_tgo_roads)
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/07
################################################################################

import geopandas as gpd
import numpy as np

FILE_PATH = "./data/22_osm_roads.gpkg"


def recategorize_osm_roads(df):
    """
    Turns osm cat. into new ones. See doc. here
    https://www.istat.it/wp-content/uploads/2021/05/Open_Street_Map_Road_Accidents_Year_2017.pdf
    p.5.
    """

    dict = {
        # Catégorie 1 – Petits chemins non motorisés Routes tertiairees / accès
        "path": "tertiaire",
        "footway": "tertiaire",
        "cycleway": "tertiaire",
        "steps": "tertiaire",
        "pedestrian": "tertiaire",
        "bridleway": "tertiaire",
        "track": "tertiaire",
        "track_grade1": "tertiaire",
        "track_grade2": "tertiaire",
        "track_grade3": "tertiaire",
        "track_grade4": "tertiaire",
        "track_grade5": "tertiaire",
        "unclassified": "tertiaire",
        "residential": "tertiaire",
        "service": "tertiaire",
        "living_street": "tertiaire",
        "tertiary": "tertiaire",
        "tertiary_link": "tertiaire",

        # Catégorie 2 – Voirie secondaire
        "secondary": "secondaire",
        "secondary_link": "secondaire",

        # Catégorie 3 – Grands axes
        "primary": "primaire",
        "primary_link": "primaire",
        "trunk": "primaire",
        "trunk_link": "primaire",
    }

    df["road_type"] = df["fclass"].map(dict)

    return df


def preprocess_osm_road(path=FILE_PATH):

    df = gpd.read_file(path, columns=["osm_id", "width", "fclass", "geometry"])

    df_categorized = recategorize_osm_roads(df)

    return df_categorized


if __name__ == "__main__":


    df = preprocess_osm_road(FILE_PATH)
