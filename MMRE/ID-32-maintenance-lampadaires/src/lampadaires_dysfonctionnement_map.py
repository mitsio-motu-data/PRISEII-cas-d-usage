################################################################################
# Script Name  : lampadaires_dysfonctionnement_map.py
# Description  : Edition d'une carte pour le policy brief
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/23
################################################################################

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from preprocessing_lampadaires import preprocess_lampadaires_prise
from lampadaires_grappes_dysfonctionnement import find_grappes

PATH_FILE_ADMIN = "./data/derived/limites_admin.gpkg"


df_prise = preprocess_lampadaires_prise(keep_comments=False)
df_grappe, _ = find_grappes()


# On ne garde que les lampadaires dysfonctionnels hors grappe
id_grappes = df_grappe.id.to_numpy()
df_prise["in_grappe"] = np.isin(df_prise["id"], id_grappes)


df_hors_grappe = df_prise[df_prise["in_grappe"] == False].copy()
df_pref = gpd.read_file(PATH_FILE_ADMIN, layer="prefectures")

# certains lampadaires sont en na (pas d'info sur le fonctionnement)
# on ne veut pas biaiser la carte
df_hors_grappe["est_fonctionnel"] = df_hors_grappe["est_fonctionnel"].fillna(0.5)  


df_carte = df_hors_grappe.groupby('prefecture')\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)


df_carte["%_dysf"] =  100 * (df_carte["count"] - df_carte["sum"]) / df_carte["count"]

df_carte = df_carte.merge(
    df_pref[["prefecture_nom", "geometry"]],
    right_on="prefecture_nom",
    left_index=True,
    how="left")

gdf_carte = gpd.GeoDataFrame(df_carte, geometry="geometry")
gdf_carte = gdf_carte.set_crs(df_pref.crs)


gdf_carte[["%_dysf", "geometry"]].plot(
    column="%_dysf",
    cmap="viridis",
    legend=True,
    #legend_kwds={'label': "Pourcentage de lampadaires dysfonctionnels (%)", 'orientation': "horizontal"},
    figsize=(10, 10),
)

# legend as xx %
plt.title("% de lampadaires dysfonctionnels (hors grappe)")

plt.axis("off")