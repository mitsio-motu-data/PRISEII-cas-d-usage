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
from statsmodels.stats.proportion import proportions_ztest
import numpy as np 

# limites admin
PATH_FILE_ADMIN = "./data/derived/limites_admin.gpkg"
PATH_FILE_URBAIN = "./data/derived/aires_urbaines.gpkg"

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


print("\nTest H0 : p1=p2")

n_dys = temp["sum"].values
n_obs = temp['count'].values

stat, pval = proportions_ztest(n_dys, n_obs)
print(f"z = {stat:.2f}, p = {pval:.4f}")

print(">>> Le taux de dysfonctionnement solaire / réseaux est le même")

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
    cmap="RdYlGn_r",
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

print(">>> Préfectures prioritaires")
################################################################################

print("-"*100+"\n4. Statistiques par types d'ampoules\n"+"-"*100)


temp = df_lampa.groupby(["ampoules_type"])\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)

print(temp)

print("Test H0 : p_led = p_hps:")

n_dys = temp.loc[["HPS/LPS", "LED"], "sum"]
n_dys = temp.loc[["HPS/LPS", "LED"], "count"]

stat, pval = proportions_ztest(n_dys, n_obs)
print(f"z = {stat:.2f}, p = {pval:.4f}")



temp = df_lampa.groupby(["ampoules_type", "type"])\
                    .agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)

print(temp)

print("Test HPS/LPS H0 : p_sol = p_réseaux:")

types = [("HPS/LPS", "Solaire"), ("HPS/LPS", "Réseaux")]

n_dys =  np.array([temp.loc[t][["sum"]].values for t in types]).flatten()
n_obs = np.array([temp.loc[t][["count"]].values for t in types]).flatten()
stat, pval = proportions_ztest(n_dys, n_obs)
print(f"z = {stat:.2f}, p = {pval:.4f}")

print("Test LED H0 : p_sol = p_réseaux")

types = [("LED", "Solaire"), ("LED", "Réseaux")]

n_dys =  np.array([temp.loc[t][["sum"]].values for t in types]).flatten()
n_obs = np.array([temp.loc[t][["count"]].values for t in types]).flatten()

stat, pval = proportions_ztest(n_dys, n_obs)
print(f"z = {stat:.2f}, p = {pval:.4f}")

print("Test Réseaux H0 : p_led = p_hps")

types = [("HPS/LPS", "Réseaux"), ("LED", "Réseaux")]
n_dys =  np.array([temp.loc[t][["sum"]].values for t in types]).flatten()
n_obs = np.array([temp.loc[t][["count"]].values for t in types]).flatten()

stat, pval = proportions_ztest(n_dys, n_obs)
print(f"z = {stat:.2f}, p = {pval:.4f}")


print(">>> Plus de dysfonctionnement pour les sodium que les LED, surtout pour les " \
"lampadaires réseaux")
################################################################################

print("-"*100+"\n5. Age du parc\n"+"-"*100)

df_lampa_year = df_lampa[df_lampa["annee"] != "Nsp"].copy()
df_lampa_year["annee"] = df_lampa_year["annee"].astype(int)


print(f"% de lampadaires dont on connait l'âge : {len(df_lampa_year) / len(df_lampa):.2%}")

print(df_lampa_year.groupby("type").annee.describe())

print(">>> Les réseaux sont plus vieux, manque de données on ne peut pas tirer grand chose")


################################################################################

print("-"*100+"\n6. Urbain vs rural\n"+"-"*100)

df_urbain = gpd.read_file(PATH_FILE_URBAIN)

# crs métrique
df_lampa.to_crs("EPSG:32631", inplace=True)
df_urbain.to_crs("EPSG:32631", inplace=True)

df_lampa["est_urbain"] = gpd.sjoin(
    df_lampa, 
    df_urbain[["geometry"]],  # On garde que la géométrie pour la jointure
    how="left", 
    predicate="within"  # ou "intersects" si tu veux aussi ceux en bordure
)["index_right"].notnull()

temp = df_lampa.groupby("est_urbain").agg({
                        "est_fonctionnel": ["count", "sum"],   
                    }).droplevel(0, axis=1)

temp["%_dysf"] = round(100 * (temp["count"] -temp["sum"]) / temp["count"], 2)

print(temp)

print(">>> Taux de dysfonctionnement comparables en aire urbaine et rurale")