################################################################################
# Script Name  : preprocessing.py
# Description  : Retravail fichier tinga pour power bi
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/28
################################################################################

import geopandas as gpd
import pandas as pd
import subprocess


def map_raccordables_to_region(df_raccordables, df_region):
    """
    Associe une region à chaque zone raccordable, arbitrairement on choisit en 
    cas de chevauchement 
    """

    join = gpd.sjoin_nearest(df_raccordables, df_region[["region_x", "geometry"]], how="left")


    df_raccordables = join.drop_duplicates(subset="geometry")

    return df_raccordables.drop(columns="index_right")


def preprocess_raccordables(df_region):
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

    df_raccordable = map_raccordables_to_region(gdf_polygons, df_region)

    return df_raccordable

def preprocess_agences_to_region():

    df_agence = gpd.read_file("./data/agence_resultat_joint3.gpkg")

    df_region = df_agence.dissolve(by="region_x", aggfunc="sum")

    cols_to_keep = ["geometry"] + [col for col in df_region if col.startswith("eligible")]

    df_region = df_region[cols_to_keep]

    # Nouvelle catégorie
    df_region["Raccordable"] = df_region["eligible_raccordable"]
    df_region["Densification"] = df_region["eligible_dens_urbain"] +\
                            df_region["eligible_dens_rural"] +\
                            df_region["eligible_ext_urbain"]
    df_region["Extension"] = df_region["eligible_ext_rural"]
    df_region["Solution off-grid"] = df_region['eligible_decentralise']

    df_region.drop(columns=[col for col in df_region if col.startswith("eligible")], 
                   inplace=True)
    
    # Régions mal écrites
    df_region.reset_index(inplace=True)
    df_region["region_x"] = df_region["region_x"].str.capitalize()

    return df_region

def map_region(df):
    # Assigne 'Grand Lomé' si la préfecture est Golfe ou Agoè-Nyivé
    df.loc[df["prefecture_nom"].isin(["Golfe", "Agoè-Nyivé"]), "region_nom"] = "Lome"

    return df


def preprocess_hexa(df_region):
    """
    Assigne une region à quasi chaque hexagone
    """
    
    df_hexa = gpd.read_file("./data/hexagones_classifies_2.gpkg")[[
        "hex_id",
        "classification",
        "pop_rgph_point_2024",
        "region_nom",
        "prefecture_nom",
        "geometry"
    ]]
    
    df_hexa = map_region(df_hexa)

    #join = gpd.sjoin(df_hexa, df_region[["region_x", "geometry"]])
    df_join = df_hexa.merge(df_region[["region_x"]], how='left', left_on='region_nom', right_on="region_x")

    # 1% des hex sur les bords
    #join = join.dropna(subset=["agence_id"]).drop(columns="index_right")
    
    return df_join

def read_data():    

    df_region = preprocess_agences_to_region()

    df_raccordable = preprocess_raccordables(df_region)

    df_hexa = preprocess_hexa(df_region)

    #df_region.drop("geometry", inplace=True)
    df_region = df_region.melt(id_vars="region_x",
                    var_name="type",
                    value_name="eligible")
    
    return df_hexa, df_raccordable, df_region


def dissolve_hexagones(df_hexa):
    """
    Rassembler les hexagone similaires de la même région
    """

    # petit buffer
    df_hexa["geometry"] = df_hexa["geometry"].buffer(1)

    df_hexa_dissolved = df_hexa.dissolve(
            by=["classification", "region_nom"],
            aggfunc={
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

    df_classif["id"] = df_classif.index
    return df_classif

def recategorize(df_classif):
    mapping_new_cat = {
        'raccordable': 'Raccordable',
        'densification_rurale': 'Densification',
        'densification_urbaine': 'Densification',
        'extension_rurale': 'Extension',
        'extension_urbaine': 'Densification',
        'zone_eloignee': 'Solution off-grid'
    }

    df_classif["new_cat"] = df_classif["classification"].map(mapping_new_cat)

    df_classif["region"] = df_classif["region_nom"].fillna("") + df_classif["region_x"].fillna("")

    df_classif = df_classif.drop(columns=["region_nom", "region_x"])

    return df_classif


def main():

    df_hexa, df_raccordable, df_region = read_data()

    df_hexa_dissolved = dissolve_hexagones(df_hexa)
    df_classif = reclassify(df_hexa_dissolved, df_raccordable)
    df_classif = recategorize(df_classif)

    return df_classif, df_region


if __name__ == "__main__":

    df_classif, df_region = main()

    df_classif.to_crs("EPSG:4326").to_file("./data/tinga.geojson")

    df_classif.drop(columns="geometry").to_csv('./data/tinga.csv', index=False)

    df_region\
        .fillna(0).to_csv("./data/resultats_par_region.csv",
                          index=False,
                          decimal=",",
                          sep=";")

    command = "tippecanoe -o  ./data/electrification_togo.mbtiles --force --drop-densest-as-needed -z 14 -Z 0 ./data/tinga.geojson"
    subprocess.run(command, shell=True)