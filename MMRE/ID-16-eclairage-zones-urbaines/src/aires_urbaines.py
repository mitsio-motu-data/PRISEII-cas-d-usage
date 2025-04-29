################################################################################
# Script Name  : aires_urbaines.py
# Description  : Regroupe les aires urbaines à partir d'heaxagones classifiés
# urbains ou ruraux. Le regroupement se fait par proximité géographique avec un
# critère de distance maximale entre les hexagones.
# La méthodologie de classification, et les éventuelles modifications qui en
# découlent sont consultables ici :
# https://docs.google.com/document/d/11S8a4HvuC65NQHHCtaCIz8MOY_EwmPu9CECKikbI9a8/edit?tab=t.0
# Auteur       : Basile Desjuzeur
# Date : 2025/04/17
################################################################################
from sklearn.cluster import DBSCAN
import geopandas as gpd
import numpy as np


########### 1. Constantes ############

# fichier source
HEXAGONES_PATH= "./data/hexagones_classifies.gpkg"
COMMUNES_PATH = "./data/limites_admin.gpkg" # attention plusieurs layers
FINAL_PATH = "./data/aires_urbaines.gpkg"


# distance maximale entre les hexagones pour les regrouper
MAX_DISTANCE = 1_000  # en mètres


########### 2. Chargement et traitement des données ############

def union_classif(row):
    """
    Harmonise les classifications INSEED et KNN
    """

    if (row["final_class"] == "urbain") or (
        row["final_class_inseed"] == "URBAIN_INSEED"
    ):
        return "urbain"

    return "rural"



def cluster_aires_urbaines(hexagones_path=HEXAGONES_PATH, max_dist=MAX_DISTANCE):
    """
    Regroupe les hexagones urbains en aires urbaines 
    """

    df_urbain = gpd.read_file(hexagones_path)

    df_urbain["class_merge"] = df_urbain.apply(lambda x: union_classif(x), axis=1)

    # "final_class" ou "final_class_inseed" ou "class_merge" voir gdoc
    df_urbain = df_urbain[df_urbain["class_merge"] == "urbain"]

    df_urbain["centroid"] = df_urbain.geometry.centroid
    coords = np.array(list(df_urbain.centroid.apply(lambda point: (point.x, point.y))))

    # Clustering spatial (distance en mètres)
    db = DBSCAN(eps=max_dist, min_samples=1).fit(
        coords
    )  # les polygones sont de 500m de large
    df_urbain["cluster"] = db.labels_

    # on va regrouper les polygones par cluster, le pavage de l'espace
    # n'étant pas parfait, il peut y avoir des trous entre les polygones
    # on dilate donc les polygones avant de les fusionner
    df_urbain["geometry"] = df_urbain.geometry.apply(lambda x: x.buffer(0.5))


    df_final = df_urbain.dissolve(
        by="cluster",
        method="unary",
        aggfunc={
            "region_nom": "first",
            "prefecture_nom": "first",
            "commune_nom": "first",
            "canton_id": "first",
            "canton_nom": "first",
            "canton_nom_alt": "first",
            "tgo_general_2020": "sum",
            "pop_rgph_2020": "sum",
            "ratio": "sum",
            "pop_rgph_point": "sum",
            "pop_rgph_point_2024": "sum",
            "nombre_de_batiments": "sum",
            "surface_batie_totale": "sum",
        },
    )

    return df_final

def map_aire_urbaine_to_commune(geometry_aire_urbaine, df_communes):
    """
    Une aire urbaine peut chevaucher plusieurs communes.
    On garde la commune qui intersecte le plus l'aire urbaine.

    Retourne le nom de la commune et sa géométrie.
    """

    communes_intersect = df_communes[df_communes.intersects(geometry_aire_urbaine)]

    if communes_intersect.shape[0] == 0:
        return None
    if communes_intersect.shape[0] == 1:
        return communes_intersect.iloc[0].commune_nom

    # calcul des surfaces d'intersection
    intersections = communes_intersect.geometry.intersection(geometry_aire_urbaine)
    areas = intersections.area

    idx_local_max = areas.idxmax()
    commune_max = communes_intersect.loc[idx_local_max]

    return commune_max["commune_nom"]


def main_aires_urbaines(hexagones_path=HEXAGONES_PATH, path_admin=COMMUNES_PATH, max_dist=MAX_DISTANCE):
    """
    Pipeline complète
    """
    # Regroupe les clusters
    df_cluster = cluster_aires_urbaines(hexagones_path, max_dist)

    # Projette les géometries sur le meme référentiel métrique
    df_communes = gpd.read_file(path_admin, layer="communes")
    df_cluster.to_crs("EPSG:32631", inplace=True)
    df_communes.to_crs("EPSG:32631", inplace=True)

    df_cluster["commune"] = df_cluster["geometry"].apply(lambda x : map_aire_urbaine_to_commune(x, df_communes=df_communes))

    return df_cluster


def print_statistics(df_cluster):

    pop_togo = 9_304_000
    sup_togo = 56_785 # km2

    pop_cluster = df_cluster["pop_rgph_point_2024"].sum()
    sup_cluster = df_cluster["geometry"].area.sum() * 10**-6

    densite_moyenne = (df_cluster.pop_rgph_point_2024 / (df_cluster.geometry.area * 10**-6)).mean()

    print(f"Nombre de clusters : {len(df_cluster)}")

    print(f"Population totale : {pop_togo} ({pop_cluster/pop_togo:.2%} du total national)")
    print(f"Superficie totale : {sup_togo} ({sup_cluster/sup_togo:.2%} du total national)")
    print(f"Densité moyenne {densite_moyenne} hab/km2")
    return

if __name__ == "__main__":

    df_cluster = main_aires_urbaines()

    print_statistics(df_cluster)

    df_cluster.to_file(FINAL_PATH)