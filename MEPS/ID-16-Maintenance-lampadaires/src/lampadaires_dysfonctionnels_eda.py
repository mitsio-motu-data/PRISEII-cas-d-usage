################################################################################
# Script Name  : lampadaires_dysfonctionnels.py
# Description  : Caractérisation des lampadaires dysfonctionnels
# Auteur       : basiledesj@hotmail.fr
# Date : 2025/04/17
################################################################################
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing_lampadaires import preprocess_lampadaires, preprocess_lampadaires_prise


# limites admin
PATH_FILE_ADMIN = "./data/derived/limites_admin.gpkg"

df_lampa = preprocess_lampadaires_prise()
df_pref = gpd.read_file(PATH_FILE_ADMIN, layer="prefectures")

###### Statistiques de base #####
################################################################################

print("-"*100+"\n1. Statistiques\n"+"-"*100)

nb_lampadaires = df_lampa.shape[0]
print(f"Nombre total de lampadaires : {nb_lampadaires}")

nb_dysf = df_lampa[df_lampa["est_fonctionnel"] == 0].shape[0]
print(f"Nombre de lampadaires dysfonctionnels : {nb_dysf} ({nb_dysf/nb_lampadaires:.2%})")
print(f"Nombre de lampadaires fonctionnels : {nb_lampadaires - nb_dysf} ({(nb_lampadaires - nb_dysf)/nb_lampadaires:.2%})")
print(f"Nombre de lampadaires non renseignés : {df_lampa['est_fonctionnel'].isna().sum()} ({df_lampa['est_fonctionnel'].isna().sum()/nb_lampadaires:.2%})")

df_lampa["est_fonctionnel"] = df_lampa["est_fonctionnel"].fillna(1)
print("\n-> On considère que les lampadaires non renseignés sont fonctionnels")

################################################################################
print("-"*100+"\n2. Statistiques par type de lampadaire\n"+"-"*100)

df_lampa["type"] = df_lampa["type"].replace("Nsp", "Réseaux").replace("Informel", "Réseaux")

temp = df_lampa.groupby("type")\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)
temp["%_total_dysf"] = round(100 * (temp["count"] -temp["sum"]) / nb_dysf, 2)
temp["%_total"] = round(100 * temp["count"] / nb_lampadaires, 2)

print(temp)
print(f"NB: On considère que les lampadaires nsp ou informels sont sur réseau (1% des lampadaires)")

################################################################################

print("-"*100+"\n3. Statistiques par secteur (carte à afficher)\n"+"-"*100)

df_carte = df_lampa.groupby('prefecture')\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)


df_carte["%_dysf"] =  100 * (df_carte["count"] - df_carte["sum"]) / df_carte["count"]


df_carte = df_carte.merge(
    df_pref[["prefecture_nom", "geometry"]],
    right_on="prefecture_nom",
    left_index=True,
    how="left"
)
print('Top 4 préfectures prioritaires :')
print(df_carte.sort_values("%_dysf", ascending=False)[["prefecture_nom", "%_dysf"]].head(4))

gdf_carte = gpd.GeoDataFrame(df_carte, geometry="geometry")
gdf_carte = gdf_carte.set_crs(df_pref.crs)

gdf_carte[["%_dysf", "geometry"]].plot(
    column="%_dysf",
    cmap="OrRd",
    legend=True,
    #legend_kwds={'label': "Pourcentage de lampadaires dysfonctionnels (%)", 'orientation': "horizontal"},
    figsize=(10, 10),
)

# legend as xx %
plt.title("% de lampadaires dysfonctionnels")

plt.axis("off")

temp = df_lampa.groupby("region")\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)

print(temp)


################################################################################

print("-"*100+"\n4. Statistiques par types d'ampoules\n"+"-"*100)

temp = df_lampa.groupby("ampoules_type")\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)

print(temp)

temp.to_csv('./temp.csv')