################################################################################
# Script Name  : routes_lampadaires_urbains.py
# Description  : Garde uniquement les routes et lampadaires en aire urbaines et
# associe les lampadaires concernés à leur route de rattachement
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/25
################################################################################
"""
Pour rattacher un lampadaire à une route on prend les hypothèses suivantes :

- rayon d'éclairage d'un lampadaire : 15m
- largeur d'une route primaire : 17m
- largeur d'une route secondaire : 12m
- largeur d'une route primaire : 5m
"""
import geopandas as gpd

from preprocessing_lampadaires import preprocess_lampadaires
from preprocessing_routes import preprocess_osm_road
from aires_urbaines import main_aires_urbaines


EPSG = "EPSG:25231"

def get_data():

    df_lampa = preprocess_lampadaires()
    df_routes = preprocess_osm_road()
    df_urbain = main_aires_urbaines().reset_index()

    # convertit au même epsg
    df_lampa.to_crs(crs=EPSG, inplace=True)
    df_routes.to_crs(crs=EPSG, inplace=True)
    df_urbain.to_crs(crs=EPSG, inplace=True)

    return df_lampa, df_routes, df_urbain

def marquer_routes_urbaines(df_routes, df_urbain):
    jointure = gpd.overlay(df_routes, df_urbain, how="intersection")
    routes_urbaines_ids = jointure["osm_id"].unique()
    df_routes['est_urbain'] = df_routes["osm_id"].isin(routes_urbaines_ids)
    return df_routes

def filter_routes_urbaines(df_routes, df_urbain):

    jointure = gpd.overlay(df_routes[df_routes["est_urbain"] == 1].reset_index(),
                           df_urbain.reset_index(),
                           how="intersection")
    
        # Construction des résultats (chaque osm_id a sa propre ligne)
    results = []
    for _, row in jointure.iterrows():
        results.append(
            {
                "osm_id": row["osm_id"],  # OSM id
                "cluster": row["cluster"],  # hexagon id
                "road_type": row["road_type"],  # Le type de route
                "road_intersected": row["geometry"],  # Géométrie de l'intersection
            }
        )

    return gpd.GeoDataFrame(results, geometry="road_intersected", crs=df_urbain.crs)


def buffer_roads(row):
   # Définir les distances de buffer en fonction du type de route
    if row["road_type"] == "primaire":
        return row["geometry"].buffer(17)  # Buffer de 15m pour les routes primaires
    elif row["road_type"] == "secondaire":
        return row["geometry"].buffer(12)  # Buffer de 10m pour les routes secondaires
    elif row["road_type"] == "tertiaire":
        return row["geometry"].buffer(5)  # Buffer de 5m pour les routes tertiaires
    else:
        return row["geometry"]  # Si le type de route n'est pas spécifié, pas de buffer

def marquer_lampadaires_urbains(df_lampa, df_urbain):
    """
    Marque les lampadaires avec une colonne booléenne 'est_urbain' en fonction de leur intersection avec une zone urbaine.
    """
    jointure = gpd.overlay(df_lampa, df_urbain, how="intersection")
    lampa_urbains_idx = jointure["id"].unique()
    df_lampa['est_urbain'] = df_lampa['id'].isin(lampa_urbains_idx)
    df_lampa['id_aire_urbaine'] = jointure['cluster']

    

    return df_lampa, jointure


def buffer_lampa(df_lampa):

    df_lampa["geometry_buffered"] = df_lampa.geometry.buffer(15)

    return df_lampa

def lampa_intersect_road(df_lampa, df_routes):

    temp_lampa = gpd.GeoDataFrame(df_lampa[["id", 'geometry_buffered']], geometry="geometry_buffered", crs=EPSG)
    temp_routes = gpd.GeoDataFrame(df_routes[["osm_id", "geometry_buffered"]], geometry="geometry_buffered", crs=EPSG)
    
    df_intersection = gpd.overlay(temp_lampa, temp_routes, how="intersection")

    # Mise à jour des colonnes
    df_lampa['intersect_road'] = df_intersection['osm_id'].notna()
    df_lampa['road_id'] = df_intersection['osm_id']

    return df_lampa

def main():

    df_lampa, df_routes, df_urbain = get_data()

    # ne garde que les élèments dans une aire urbaine
    df_routes = marquer_routes_urbaines(df_routes, df_urbain)
    df_lampa = marquer_lampadaires_urbains(df_lampa, df_urbain)

    # applique des buffer
    df_routes["geometry_buffered"] = df_routes.apply(buffer_roads, axis=1)
    df_lampa = buffer_lampa(df_lampa)

    # intersecte les routes
    df_lampa = lampa_intersect_road(df_lampa, df_routes)

    return df_lampa, df_routes, df_urbain


if __name__ == "__main__":

    df_lampa, df_routes, df_urbain  = main()

    df_routes_intersec = filter_routes_urbaines(df_routes, df_urbain)

    df_lampa.drop(columns=["geometry_buffered"]).to_file("./data/main.gpkg", layer="lampadaires")
    df_routes.drop(columns=["geometry_buffered"]).to_file("./data/main.gpkg", layer="routes")
    df_urbain.to_file("./data/main.gpkg", layer="aires_urbaines")
    df_routes_intersec.to_file("./data/main.gpkg", layer="routes_urbaines")