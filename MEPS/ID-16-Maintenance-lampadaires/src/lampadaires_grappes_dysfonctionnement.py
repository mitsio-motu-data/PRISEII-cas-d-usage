################################################################################
# Script Name  : lampadaires_dysfonctionnels_causes.py
# Description  : Grappes de lampadaires dysfonctionnels
# Auteur       : basiledesj@hotmail.fr
# Date : 2025/04/17
################################################################################
"""
On parle d'un effet de grappe lorsque plus de 4 lampadaires proches, typiquement
sur la même rue, sont dysfonctionnels et sur réseaux et qu'il n'y a pas ou peu
de lampadaires fonctionnels entre eux. On va appliquer l'algorithme DBSCAN dans
l'espace latitude x longitude pour repérer ces grappes.

On adopte une approche différenciée à Lomé et en dehors, car a Lomé les lampadaires
sont exceptionnellement proches. On doit dont paramétrer différemment l'algorithme.

NB : On a tenté sans succès un clustering sur les lampadaires fonctionnels
(lat x lon x est_fonctionnel) voir commit 431cc85a4379...
"""
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
import math 

from preprocessing_lampadaires import preprocess_lampadaires_prise

############ lecture et preprocessing
df_prise = preprocess_lampadaires_prise(keep_comments=True)

# projette sur un référentiel métrique
df_prise.to_crs("EPSG:32631", inplace=True)

nb_dysfonctionnels = df_prise[df_prise["est_fonctionnel"] == 0].shape[0]
df_prise.loc[:, "x"], df_prise.loc[:, "y"] = df_prise.geometry.x, df_prise.geometry.y


# ne garde que les lampadaires dysfonctionnels sur réseaux
df_prise = df_prise[df_prise["type"] == "Réseaux"].copy()
df_prise = df_prise[df_prise["est_fonctionnel"] == 0].copy()


def cluster_and_filter(df, eps, min_cluster_size):
    """
    Applique DBSCAN et filtre les petits clusters.
    
    Args:
        df (GeoDataFrame): les lampadaires à traiter (doit contenir 'x' et 'y')
        eps (float): distance maximale pour DBSCAN
        min_cluster_size (int): taille minimale d’un cluster à conserver

    Returns:
        GeoDataFrame: lampadaires appartenant à un cluster valide
    """
    X = df[["x", "y"]].to_numpy()

    db = DBSCAN(eps=eps, min_samples=1, n_jobs=8).fit(X)
    df = df.copy()
    df["cluster"] = db.labels_

    # enlever les bruits
    df = df[df["cluster"] != -1]

    # garder les clusters significatifs
    valid_clusters = df["cluster"].value_counts()
    valid_clusters = valid_clusters[valid_clusters > min_cluster_size].index
    df = df[df["cluster"].isin(valid_clusters)]

    return df

####### Approche différenciée Lomé vs reste
df_lome, df_hors_lome = df_prise[df_prise["prefecture"] == "Golfe"],\
                        df_prise[df_prise["prefecture"] != "Golfe"]    

# Paramètres
params = [
    {"df": df_lome, "eps": 45,  "min_size": 5},
    {"df": df_hors_lome, "eps": 110, "min_size": 4}
]

# Traitement
df_final = gpd.GeoDataFrame(pd.concat([
    cluster_and_filter(p["df"], p["eps"], p["min_size"])
    for p in params
], ignore_index=True))

######### Statistiques ##########

print("Nombre de grappes : ", len(df_final["cluster"].unique()))
print(
    f"Nombre de lampadaires dans une grappe: {len(df_final)} soit {100 * len(df_final) / nb_dysfonctionnels:.2f}% des lampadaires dysfonctionnels"
)
print(
    f"Nombre moyen de lampadaires par grappes : {len(df_final) / len(df_final['cluster'].unique()):.2f}"
)


# get the clusters with the most lampadaires
threshold = df_final.groupby("cluster").size().quantile(0.95)
# arrondi au mutplide de 5 inférieur pour lisibilité
threshold = math.floor(threshold / 5) * 5
big_clusters = df_final.groupby("cluster").size()
big_clusters = big_clusters[big_clusters > threshold].index.tolist()

df_final["is_big_cluster"] = df_final["cluster"].isin(big_clusters)

nb_lampadaires_in_big_clusters = df_final[df_final["is_big_cluster"]].shape[0]

print(
    f"Nombre de 'méga-grappes' : {len(big_clusters)} "
    f"(>{threshold:.0f} lampadaires), "
    f"soit {100 * nb_lampadaires_in_big_clusters / nb_dysfonctionnels:.2f}% "
    f"des lampadaires dysfonctionnels"
)


df_final.to_file("./data/derived/grappes_lampadaires_prise.gpkg")
