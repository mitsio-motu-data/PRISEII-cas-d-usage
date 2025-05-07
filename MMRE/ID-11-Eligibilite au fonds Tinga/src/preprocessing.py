################################################################################
# Script Name  : preprocessing.py
# Description  : Prétraitement du fichier Tinga pour Power BI
# Auteur       : basile@mitsiomotu.com
# Date         : 2025/04/28
################################################################################

import geopandas as gpd
import pandas as pd
import subprocess
import os

# =============================== Lecture & Nettoyage ===============================

def load_agences() -> gpd.GeoDataFrame:
    return gpd.read_file("./data/agence_resultat_joint3.gpkg")


def load_raccordables() -> gpd.GeoDataFrame:
    df = gpd.read_file("./data/zone_1_raccordable.gpkg")[["geometry"]]
    multi = df.geometry[0]
    geometries = list(multi.geoms)
    return gpd.GeoDataFrame(geometry=geometries, crs=df.crs)


def load_hexagones() -> gpd.GeoDataFrame:
    return gpd.read_file("./data/hexagones_classifies_2.gpkg")[[
        "hex_id", "classification", "pop_rgph_point_2024",
        "region_nom", "prefecture_nom", "geometry"
    ]]


# ============================== Traitements Régionaux ==============================

def clean_region_names(df: pd.DataFrame) -> pd.DataFrame:
    df["region_x"] = df["region_x"].str.capitalize()
    return df


def aggregate_regions(df_agences: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    agg_dict = {
    col: "first" if col == "region_x" else "sum"
    for col in df_agences.columns
    if col != "geometry"
    }

    df = df_agences.dissolve(by="agence", aggfunc=agg_dict) 

    cols = ["geometry", "region_x"] + [col for col in df.columns if col.startswith("eligible")]
    df = df[cols]

    df["Raccordable"] = df["eligible_raccordable"]
    df["Densification"] = (
        df["eligible_dens_urbain"] +
        df["eligible_dens_rural"] +
        df["eligible_ext_urbain"]
    )
    df["Extension"] = df["eligible_ext_rural"]
    df["Solution off-grid"] = df["eligible_decentralise"]

    df.drop(columns=[col for col in df.columns if col.startswith("eligible")], inplace=True)
    df = df.reset_index()
    return clean_region_names(df)


# ============================ Assignations Spatiales ===============================

def assign_region_to_raccordables(df_raccordables: gpd.GeoDataFrame,
                                  df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin_nearest(df_raccordables, df_regions[["region_x", "geometry"]], how="left")
    return joined.drop_duplicates(subset="geometry").drop(columns="index_right")


def assign_region_to_hexagones(df_hexa: gpd.GeoDataFrame,
                               df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df_hexa = df_hexa.copy()
    df_hexa.loc[df_hexa["prefecture_nom"].isin(["Golfe", "Agoè-Nyivé"]), "region_nom"] = "Lome"
    return df_hexa.merge(df_regions[["region_x"]], how="left", left_on="region_nom", right_on="region_x")


# ============================= Regroupements & Fusion ==============================

def dissolve_hexagones(df_hexa: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df_hexa.copy()
    df["geometry"] = df["geometry"].buffer(1)
    return df.dissolve(
        by=["classification", "region_nom"],
        aggfunc={"pop_rgph_point_2024": "sum"}
    ).reset_index()


def reclassify_zones(df_hexa_diss: gpd.GeoDataFrame,
                     df_raccordables: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df_restants = gpd.overlay(df_hexa_diss, df_raccordables, how="difference")
    df_raccordables = df_raccordables.copy()
    df_raccordables["classification"] = "raccordable"
    df_all = pd.concat([df_raccordables, df_restants], ignore_index=True)
    df_all["id"] = df_all.index
    return df_all


def recategorize(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mapping = {
        'raccordable': 'Raccordable',
        'densification_rurale': 'Densification',
        'densification_urbaine': 'Densification',
        'extension_rurale': 'Extension',
        'extension_urbaine': 'Densification',
        'zone_eloignee': 'Solution off-grid'
    }
    df = df.copy()
    df["new_cat"] = df["classification"].map(mapping)
    df["region"] = df["region_nom"].fillna("") + df["region_x"].fillna("")
    return df.drop(columns=["region_nom", "region_x"])

def simplify_geometries_inplace(df_classif, tol=0.01):
    """
    Explose les multipolygones, simplifie les géométries dans df_classif pour les catégories
    'Solution off-grid' et 'Extension', et les dissout par catégorie.

    Modifie df_classif en place et retourne la version modifiée.

    Args:
        df_classif (GeoDataFrame): GeoDataFrame contenant 'new_cat' et 'geometry'.
        tol (float): Tolérance de simplification.

    Returns:
        GeoDataFrame: Le df_classif modifié avec géométries simplifiées et fusionnées.
    """
    # Filtrage des catégories ciblées
    mask = df_classif["new_cat"].isin(["Solution off-grid", "Extension"])
    df_target = df_classif[mask].copy()

    # Explosion des multipolygones
    df_target = df_target.explode(index_parts=False)

    # Simplification
    df_target["geometry"] = df_target.geometry.simplify(tolerance=tol, preserve_topology=True)

    # Dissolution par catégorie
    #df_target = df_target.dissolve(by="new_cat", as_index=False)

    # Suppression des anciennes lignes et remplacement par les nouvelles
    df_classif = df_classif[~mask]  # on garde le reste tel quel
    df_classif = pd.concat([df_classif, df_target], ignore_index=True)

    df_classif["id"] = df_classif.index
    return df_classif

# =============================== Pipeline Principal ================================

def preprocess_all():
    df_agences = load_agences()
    df_regions = aggregate_regions(df_agences)

    df_raccordables_raw = load_raccordables()
    df_raccordables = assign_region_to_raccordables(df_raccordables_raw, df_regions)

    df_hexagones_raw = load_hexagones()
    df_hexagones = assign_region_to_hexagones(df_hexagones_raw, df_regions)

    df_regions_melted = df_regions.melt(id_vars="region_x",
                                        var_name="type",
                                        value_name="eligible")
    
    df_regions_melted = df_regions_melted.iloc[6:, ]

    df_hexa_diss = dissolve_hexagones(df_hexagones)
    df_classif = reclassify_zones(df_hexa_diss, df_raccordables)
    df_classif = recategorize(df_classif)
    df_classif = simplify_geometries_inplace(df_classif)

    return df_classif, df_regions_melted


# ============================== Export & Finalisation ==============================

def export_all(df_classif: gpd.GeoDataFrame, df_regions: pd.DataFrame):
    df_classif.to_crs("EPSG:4326").to_file("./data/tinga.geojson")
    df_classif.drop(columns="geometry").to_csv('./data/tinga.csv', index=False)

    df_regions.fillna(0).to_csv("./data/resultats_par_region.csv",
                                index=False, decimal=",", sep=";")

    subprocess.run(
        "tippecanoe -o ./data/electrification_togo.mbtiles "
        "--force --drop-densest-as-needed -z 14 -Z 0 ./data/tinga.geojson",
        shell=True
    )
    os.remove("./data/tinga.geojson")



def main():
    df_classif, df_regions = preprocess_all()
    export_all(df_classif, df_regions)


if __name__ == "__main__":
    main()
