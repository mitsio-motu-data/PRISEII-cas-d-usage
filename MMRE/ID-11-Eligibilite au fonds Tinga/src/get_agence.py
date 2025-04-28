################################################################################
# Script Name  : get_agence.py
# Description  : Attribuer à chaque hexagone l'agence en fonction de la plus grande intersection
# Auteur       : basile@mitsiomotu.com
# Date         : 2025/04/28
################################################################################

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Lecture des données
df_agence = gpd.read_file("./data/agence_resultats_joint.gpkg")
df_agence["agence_id"] = df_agence.index

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

def build_kdtree(geometries):
    """Construit un KDTree à partir des centroïdes des géométries."""
    centroids = geometries.centroid
    coords = np.array(list(zip(centroids.x, centroids.y)))
    return cKDTree(coords), centroids

def get_candidate_pairs(df_hexa, df_agence, k=1):
    """Associe chaque hexagone à ses k agences les plus proches spatialement."""
    tree, _ = build_kdtree(df_agence.geometry)
    hexa_coords = np.array(list(zip(df_hexa.geometry.centroid.x, df_hexa.geometry.centroid.y)))

    distances, indices = tree.query(hexa_coords, k=k)

    candidates = []
    for idx_hexa, idx_agences in enumerate(indices):
        if k == 1:
            idx_agences = [idx_agences]
        for idx_agence in idx_agences:
            candidates.append({
                "hexa_idx": idx_hexa,
                "agence_idx": idx_agence
            })

    return pd.DataFrame(candidates)

def compute_intersections(candidates_df, df_hexa, df_agence):
    """Calcule les surfaces d'intersection entre chaque hexagone et ses agences candidates."""
    intersections = []

    for _, row in candidates_df.iterrows():
        geom_hexa = df_hexa.iloc[row["hexa_idx"]].geometry
        geom_agence = df_agence.iloc[row["agence_idx"]].geometry
        inter = geom_hexa.intersection(geom_agence)
        if not inter.is_empty:
            intersections.append({
                "hex_id": df_hexa.iloc[row["hexa_idx"]].hex_id,
                "agence_id": df_agence.iloc[row["agence_idx"]].agence_id,
                "surface_intersection": inter.area
            })

    return pd.DataFrame(intersections)

def assign_missing_by_nearest(df_classif, df_agence):
    """Assigne une agence aux hexagones restants via nearest neighbor."""
    missing = df_classif[df_classif["agence_id"].isna()]
    if missing.empty:
        return df_classif

    print(f"{len(missing)} hexagones sans attribution. Assignation par nearest neighbor...")

    tree, _ = build_kdtree(df_agence.geometry)
    missing_coords = np.array(list(zip(missing.geometry.centroid.x, missing.geometry.centroid.y)))

    _, nearest_idx = tree.query(missing_coords, k=1)
    nearest_agences = df_agence.iloc[nearest_idx]["agence_id"].values

    df_classif.loc[df_classif["agence_id"].isna(), "agence_id"] = nearest_agences

    return df_classif

def merge_agences(df_classif, df_agence, buffer_distance=30, k=1):
    """
    Attribue à chaque hexagone l'agence correspondant à la plus grande intersection.
    """

    df_classif = df_classif.copy()
    df_agence = df_agence.copy()
    df_agence["geometry"] = df_agence["geometry"].buffer(buffer_distance)

    # Trouver les paires hexagone/agence candidates
    candidates_df = get_candidate_pairs(df_classif, df_agence, k=k)

    # Calculer les surfaces d'intersection
    intersections_df = compute_intersections(candidates_df, df_classif, df_agence)

    if intersections_df.empty:
        raise ValueError("Aucune intersection trouvée.")

    # Choisir la meilleure agence pour chaque hexagone
    idx_max = intersections_df.groupby("hex_id")["surface_intersection"].idxmax()
    best_matches = intersections_df.loc[idx_max, ["hex_id", "agence_id"]]

    # Joindre les meilleures correspondances
    df_classif = df_classif.merge(best_matches, on="hex_id", how="left")

    # Traiter les hexagones sans correspondance
    df_classif = assign_missing_by_nearest(df_classif, df_agence)

    # Nettoyage final
    if "hex_id" in df_classif.columns:
        df_classif = df_classif.drop(columns=["hex_id"])

    return df_classif

# Exécution
df = merge_agences(df_hexa, df_agence)
