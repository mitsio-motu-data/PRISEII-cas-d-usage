import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import pandas as pd
import geopandas as gpd
import h3pandas
import networkx as nx
import osmnx as ox
from scipy.spatial import KDTree
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from scipy.optimize import minimize
import numpy as np
import random
import contextily as ctx
import time

############################################################################################################
pop=gpd.read_file("./Pop_Togo/Pop_density_map_Meta/grille_200m_togo.gpkg")
Centres_sante=gpd.read_file("./Donnees_geoportail/sant_etablissements_sante.gpkg")
Admin_sante=gpd.read_file("./Donnees_geoportail/admin_sante.gpkg")
# Map

commune=gpd.read_file("./map/14_11_22_Communes_du_Togo_2.gpkg")
canton=gpd.read_file("./map/14_11_22_Togo_Cantons2_4326.gpkg")
prefecture=gpd.read_file("./map/prefectures.gpkg")
region=gpd.read_file("./map/regions_Togo.gpkg")
Lome=gpd.read_file("./map/2024-05-31_Grand_Lomé.gpkg")

# Hexagones_classifies
hexagones_classifies=gpd.read_file("./map/hexagones_classifies.gpkg")
hexagones_classifies=hexagones_classifies[['final_class','geometry']]

# Conversion en mètre
pop=pop.to_crs(epsg=32631)
Centres_sante=Centres_sante.to_crs(epsg=32631)
Centres_sante_publics=Centres_sante[Centres_sante['secteur']!='Privé']
Liste_retenue=['USP 2','USP 1','Hopital 1','Hopital 2','CHR','Hopital Spécialisé']
Centres_sante_publics_filtre=Centres_sante_publics[Centres_sante_publics['etablissement_type'].isin(Liste_retenue)]
Admin_sante=Admin_sante.to_crs(epsg=32631)

commune=commune.to_crs(epsg=32631)
canton=canton.to_crs(epsg=32631)
prefecture=prefecture.to_crs(epsg=32631)
region=region.to_crs(epsg=32631)
hexagones_classifies=hexagones_classifies.to_crs(epsg=32631)
Lome=Lome.to_crs(epsg=32631)
prefecture_tot=pd.concat([prefecture[['prefecture_nom','region_nom','geometry']], Lome[['prefecture_nom','region_nom','geometry']]], axis=0)
prefecture=prefecture_tot.reset_index(drop=True)


### Pref without agoè-Nyivé & Golfe
pref_cp=prefecture.copy()
pref_cp=pref_cp[~pref_cp['prefecture_nom'].isin(['Agoè-Nyivé', 'Golfe'])]
pref_cp = pref_cp.dissolve(by='region_nom', as_index=False)
region_sans_lome=pref_cp[['region_nom','geometry']]
Lome['region_nom']='Lome'
region_avec_lome=gdf_concat = pd.concat([region_sans_lome, Lome[['region_nom','geometry']]], ignore_index=True)
region_avec_lome=gpd.GeoDataFrame(region_avec_lome,geometry='geometry',crs=region.crs)
############################################################################################################
#
#   Typologie de la couverture sante au Togo
#
############################################################################################################
marker_list = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>','d','^','h']
cmap_marker=plt.cm.get_cmap('hsv',Centres_sante_publics['etablissement_type'].nunique())
fig, ax = plt.subplots(figsize=(15, 15))
region.plot(ax=ax,facecolor="None",edgecolor='black')
for ind,element in enumerate(Centres_sante_publics['etablissement_type'].unique()):
    Centres_sante_publics[Centres_sante['etablissement_type']==element].plot(ax=ax,color=cmap_marker(ind),marker=marker_list[ind],markersize=25,label=element)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  
ax.set_title("Typologie des centres de santé")
plt.show()

############################################################################################################
#
#   Population
#
############################################################################################################
gpd_pop = gpd.sjoin(pop, region, predicate="within", how="inner")
gpd_pop=gpd_pop.rename(columns={'total_pop_meta':'population'})
gpd_pop=gpd_pop[['population','geometry']]
gpd_pop=gpd_pop[gpd_pop['population']>0.001]

############################################################################################################
#
#   Calcul distance minimales
#
############################################################################################################
from sklearn.neighbors import BallTree
start_time = time.time()
coords_sante = np.array(list(zip(Centres_sante_publics_filtre.geometry.x, Centres_sante_publics_filtre.geometry.y)))
tree = BallTree(coords_sante, leaf_size=15, metric='euclidean')
# Points à tester
centroids = gpd_pop.geometry.centroid
coords_pop = np.array(list(zip(centroids.x, centroids.y)))
# Trouver les plus proches voisins
distances, indices = tree.query(coords_pop, k=1)
# Remplir la colonne
gpd_pop['distance_soin_proche'] = distances
end_time= time.time()
print(f"Elapsed time computing minimal distance : {end_time-start_time} secondes")
  
###########################################################################################################
# Aggragation sur différente échelles
def Agg_zone(gdf,zone,zone_name):
    if 'centroid' not in gdf.columns:  
        gdf['centroid'] = gdf['geometry'].centroid
    gdf=gdf.set_geometry('centroid')
    gdf=gdf.reset_index()
    pop_par_zone = gpd.sjoin(gdf, zone, predicate='within', how='left')
    pop_par_zone['weighted_distance'] = pop_par_zone['population'] * pop_par_zone['distance_soin_proche']
    agg_zone = pop_par_zone.groupby(zone_name).agg(
        weighted_sum=('weighted_distance', 'sum'),
        total_population=('population', 'sum'),
        zone_name=(zone_name, 'first')
    )
    agg_zone['weighted_avg_distance'] = agg_zone['weighted_sum'] / agg_zone['total_population']
    agg_zone = agg_zone.merge(zone, on=zone_name, how='inner')
    agg_zone=gpd.GeoDataFrame(agg_zone,geometry='geometry',crs=gdf.crs)
    return agg_zone

def print_agg_distance_over_zone(mean_agg,screen,cmap_bounds="None"):
    if screen == 'laptop':
        Fontsize_title=14
        Fontsize_label=12
        Fontsize_tick_cbar=10
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["weighted_avg_distance"].min(), mean_agg["weighted_avg_distance"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="weighted_avg_distance",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Distance moyenne au centre de\nsanté le plus proche",  # texte
                transform=ax.transAxes,         # coordonnées relatives à l'axe (0–1)
                ha='center', va='bottom',       # alignement horizontal / vertical
                fontsize=Fontsize_title)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Obligatoire pour éviter une erreur
        cax = inset_axes(ax,
                        width="5%",       # % de la largeur de l'axe parent
                        height="50%",     # % de la hauteur de l'axe parent
                        loc='lower left',
                        bbox_to_anchor=(1.05, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Distance moyenne (m)",fontsize=Fontsize_label)
        cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
        rect = Rectangle(
            (0.41, 0.115), 0.26, 0.81,                        # position et taille (x, y, width, height)
            linewidth=2,                        # épaisseur du trait
            edgecolor='black',                  # couleur de bord
            facecolor='none',                   # transparent à l'intérieur
            transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
            zorder=1000                         # bien au-dessus de tout le reste
        )
        fig.patches.append(rect)
        plt.show()
    else :   
        Fontsize_title=20
        Fontsize_label=18
        Fontsize_tick_cbar=16
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["weighted_avg_distance"].min(), mean_agg["weighted_avg_distance"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="weighted_avg_distance",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Distance moyenne au centre de\nsanté le plus proche",  # texte
                transform=ax.transAxes,         # coordonnées relatives à l'axe (0–1)
                ha='center', va='bottom',       # alignement horizontal / vertical
                fontsize=Fontsize_title)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Obligatoire pour éviter une erreur
        cax = inset_axes(ax,
                        width="5%",       # % de la largeur de l'axe parent
                        height="50%",     # % de la hauteur de l'axe parent
                        loc='lower left',
                        bbox_to_anchor=(1.05, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Distance moyenne (m)",fontsize=Fontsize_label)
        cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
        rect = Rectangle(
            (0.36, 0.115), 0.4, 0.81,                        # position et taille (x, y, width, height)
            linewidth=2,                        # épaisseur du trait
            edgecolor='black',                  # couleur de bord
            facecolor='none',                   # transparent à l'intérieur
            transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
            zorder=1000                         # bien au-dessus de tout le reste
        )
        fig.patches.append(rect)
        plt.show()


mean_dist_region=Agg_zone(gpd_pop.copy(),region,'region_nom')
mean_dist_prefecture=Agg_zone(gpd_pop.copy(),prefecture,'prefecture_nom')
mean_dist_communes=Agg_zone(gpd_pop.copy(),commune,'commune_nom')
mean_dist_canton=Agg_zone(gpd_pop.copy(),canton,'canton_nom')
mean_dist_region_avec_lome=Agg_zone(gpd_pop.copy(),region_avec_lome,'region_nom')

print_agg_distance_over_zone(mean_dist_region,'pc_fixe')
print_agg_distance_over_zone(mean_dist_prefecture,'pc_fixe')
print_agg_distance_over_zone(mean_dist_communes,'pc_fixe',[0,6000])
print_agg_distance_over_zone(mean_dist_canton,'pc_fixe',[0,6000])

mean_dist_region[['region_nom','weighted_avg_distance']].to_csv("distance_moyenne_region.csv", index=False, encoding='utf-8-sig')
mean_dist_prefecture[['region_nom','prefecture_nom','weighted_avg_distance']].to_csv("distance_moyenne_prefecture.csv", index=False, encoding='utf-8-sig')
mean_dist_communes[['region_nom','prefecture_nom','commune_nom','weighted_avg_distance']].to_csv("distance_moyenne_commune.csv", index=False, encoding='utf-8-sig')
mean_dist_canton[['region_nom','prefecture_nom','commune_nom','canton_nom','weighted_avg_distance']].to_csv("distance_moyenne_canton.csv", index=False, encoding='utf-8-sig')
mean_dist_region_avec_lome[['region_nom','weighted_avg_distance']].to_csv("distance_moyenne_region_avec_lome.csv", index=False, encoding='utf-8-sig')


##########################################################################################################################
#
# Population en difficulté d'accès au soin
#
##########################################################################################################################
pop_dif_acces=gpd_pop[gpd_pop['distance_soin_proche']>5000].copy()
# Aggragation sur différente échelles
def Agg_pop_zone(gdf,zone,zone_name):
    if 'centroid' not in gdf.columns:  
        gdf['centroid'] = gdf['geometry'].centroid
    gdf=gdf.set_geometry('centroid')
    gdf=gdf.reset_index()
    pop_par_zone = gpd.sjoin(gdf, zone, predicate='within', how='left')
    agg_zone = pop_par_zone.groupby(zone_name).agg(
        total_population=('population', 'sum'),
        zone_name=(zone_name, 'first')
    )
    agg_zone = agg_zone.merge(zone, on=zone_name, how='right')
    # Remplacer les valeurs manquantes (zones sans population)
    agg_zone['total_population'] = agg_zone['total_population'].fillna(0)
    agg_zone=gpd.GeoDataFrame(agg_zone,geometry='geometry',crs=gdf.crs)
    return agg_zone

def print_pop_dif_over_zone(mean_agg,screen,cmap_bounds="None"):
    if screen=='laptop':
        Fontsize_title=14
        Fontsize_label=12
        Fontsize_tick_cbar=10
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["total_population"].min(), mean_agg["total_population"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="total_population",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Nombre de personnes en\ndifficulté d'accès",  # texte
                transform=ax.transAxes,         # coordonnées relatives à l'axe (0–1)
                ha='center', va='bottom',       # alignement horizontal / vertical
                fontsize=Fontsize_title)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Obligatoire pour éviter une erreur
        cax = inset_axes(ax,
                        width="5%",       # % de la largeur de l'axe parent
                        height="50%",     # % de la hauteur de l'axe parent
                        loc='lower left',
                        bbox_to_anchor=(1.0, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Nombre de personnes",fontsize=Fontsize_label)
        cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
        rect = Rectangle(
            (0.41, 0.115), 0.26, 0.81,                        # position et taille (x, y, width, height)
            linewidth=2,                        # épaisseur du trait
            edgecolor='black',                  # couleur de bord
            facecolor='none',                   # transparent à l'intérieur
            transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
            zorder=1000                         # bien au-dessus de tout le reste
        )
        fig.patches.append(rect)
        plt.show()
    else:
        Fontsize_title=20
        Fontsize_label=18
        Fontsize_tick_cbar=16
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["total_population"].min(), mean_agg["total_population"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="total_population",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Nombre de personnes en\ndifficulté d'accès",  # texte
                transform=ax.transAxes,         # coordonnées relatives à l'axe (0–1)
                ha='center', va='bottom',       # alignement horizontal / vertical
                fontsize=Fontsize_title)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Obligatoire pour éviter une erreur
        cax = inset_axes(ax,
                        width="5%",       # % de la largeur de l'axe parent
                        height="50%",     # % de la hauteur de l'axe parent
                        loc='lower left',
                        bbox_to_anchor=(1.0, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Nombre de personnes",fontsize=Fontsize_label)
        cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
        rect = Rectangle(
            (0.36, 0.115), 0.4, 0.81,                        # position et taille (x, y, width, height)
            linewidth=2,                        # épaisseur du trait
            edgecolor='black',                  # couleur de bord
            facecolor='none',                   # transparent à l'intérieur
            transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
            zorder=1000                         # bien au-dessus de tout le reste
        )
        fig.patches.append(rect)
        plt.show()

Pop_dif_region=Agg_pop_zone(pop_dif_acces.copy(),region,'region_nom')
Pop_dif_prefecture=Agg_pop_zone(pop_dif_acces.copy(),prefecture,'prefecture_nom')
Pop_dif_commune=Agg_pop_zone(pop_dif_acces.copy(),commune,'commune_nom')
Pop_dif_canton=Agg_pop_zone(pop_dif_acces.copy(),canton,'canton_nom')
Pop_dif_region_avec_lome=Agg_pop_zone(pop_dif_acces.copy(),region_avec_lome,'region_nom')

print_pop_dif_over_zone(Pop_dif_region,'pc_fixe')
print_pop_dif_over_zone(Pop_dif_prefecture,'pc_fixe')
print_pop_dif_over_zone(Pop_dif_commune,'pc_fixe')
print_pop_dif_over_zone(Pop_dif_canton,'pc_fixe')

Pop_dif_region[['region_nom','total_population','pop_percent']].to_csv("population_exclu_region.csv", index=False, encoding='utf-8-sig')
Pop_dif_prefecture[['region_nom','prefecture_nom','pop_percent','total_population']].to_csv("population_exclu_prefecture.csv", index=False, encoding='utf-8-sig')
Pop_dif_commune[['region_nom','prefecture_nom','commune_nom','total_population']].to_csv("population_exclu_commune.csv", index=False, encoding='utf-8-sig')
Pop_dif_canton[['region_nom','prefecture_nom','commune_nom','canton_nom','total_population']].to_csv("population_exclu_canton.csv", index=False, encoding='utf-8-sig')
Pop_dif_region_avec_lome[['region_nom','total_population']].to_csv("population_exclu_region_avec_lome.csv", index=False, encoding='utf-8-sig')

Pop_dif_prefecture['pop_percent']=100*Pop_dif_prefecture['total_population']/Pop_dif_prefecture['total_population'].sum(axis=0)
Pop_pref_percent=Pop_dif_prefecture[['region_nom','prefecture_nom','pop_percent','total_population']].sort_values(by='pop_percent',ascending=False)



gpd_pop['centroid']=gpd_pop.geometry.centroid
gpd_pop=gpd_pop.set_geometry('centroid')
gpd_pop=gpd_pop.sjoin(prefecture,predicate='within',how='left')
gpd_pop['population']=8095000*gpd_pop['population']/gpd_pop['population'].sum()
prefecture_pop=gpd_pop.groupby("prefecture_nom").agg(
    population_total_prefectoral=('population', 'sum'))

prefecture_pop['population_total_prefectoral']=8095000*prefecture_pop['population_total_prefectoral']/prefecture_pop['population_total_prefectoral'].sum()
Pop_dif_prefecture=Pop_dif_prefecture.merge(prefecture_pop,on="prefecture_nom",how='inner')
Pop_dif_prefecture['percent/pop_pref']=100*Pop_dif_prefecture['total_population']/Pop_dif_prefecture['population_total_prefectoral']

prefecture.to_file("prefecture_lancement.gpkg", driver="GPKG")
raise SystemExit
#####################################################################################################
######################################################################
# Optimisation brutale sans considération de la saturation d'USP
######################################################################
#####################################################################################################
Liste_commune=['Dankpen 1','Dankpen 2','Dankpen 3','Anié 1','Haho 2','Tchamba 1','Tchamba 3','Oti-Sud 1','Oti-Sud 2','Ogou 4']
restricted_commune=commune[commune['commune_nom'].isin(Liste_commune)]
restricted_commune=gpd.GeoDataFrame({"geometry":[restricted_commune['geometry'].union_all()]},crs=commune.crs)
def evaluate_good_spot(updated_pop):
    Find_good_spot=updated_pop.copy()
    Find_good_spot['buffer']=Find_good_spot.geometry.buffer(5000)
    Find_good_spot['geometry'] = Find_good_spot['buffer']
    Find_good_spot = Find_good_spot.set_geometry('buffer')
    pop_cumul=updated_pop.copy()
    pop_cumul['centroid'] = pop_cumul.geometry.centroid
    centroids = gpd.GeoDataFrame(pop_cumul[['population']], geometry=pop_cumul['centroid'], crs=updated_pop.crs)
    joined = gpd.sjoin(centroids, Find_good_spot, predicate='within', how='inner')
    agg_pop_large_dense = joined.groupby('index_right').agg(
        Pop=('population_left', 'sum'))
    agg_pop_large_dense.reset_index(inplace=True)
    agg_pop_large_dense.rename(columns={'index_right': 'index'}, inplace=True)   
    Find_good_spot.reset_index(inplace=True) 
    agg_pop_large_dense = agg_pop_large_dense.merge(Find_good_spot, on='index', how='inner')
    agg_pop_large_dense=agg_pop_large_dense[['Pop','buffer']]
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('buffer')
    agg_pop_large_dense['centroid']=agg_pop_large_dense.geometry.centroid
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('centroid')
    return agg_pop_large_dense


new_USP=[]
nombre_patients=[]
Optimisation={}
Tot_num_USP=20
pop=pop_dif_acces.copy()
for num_school in range(Tot_num_USP):
    Optimisation[num_school]={}
    Optimisation[num_school]['track']=[]
    Optimisation[num_school]['goal']=[]
    def objectif(x, pop,Optimisation_dict):
        population_a_optim = pop.copy()
        Optimisation_dict['track'].append(x)
        point = Point(x[0], x[1])
        gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
        gpd_buffer=gpd_buffer.buffer(5000)
        gpd_buffer.set_crs("EPSG:32631", inplace=True)
        gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
        Integ = gpd.sjoin(population_a_optim, gpd_buffer, predicate="within", how="inner")  
        Tot_integ = -Integ['population'].sum(axis=0)
        Optimisation_dict['goal'].append(Tot_integ)
        return Tot_integ
    bounds = [(0, 201200), (680000, 1248000)]
    # Point de départ
    agg_pop_large_dense=evaluate_good_spot(pop)
    Ligne=agg_pop_large_dense[agg_pop_large_dense['Pop']==agg_pop_large_dense['Pop'].max()].centroid
    x0 = [Ligne.geometry.iloc[0].x, Ligne.geometry.iloc[0].y]
    agg_pop_large_dense.drop(Ligne.index,inplace=True)
    # Appel à l’optimiseur
    # resultat = minimize(objectif, x0, args=(pop), method='SLSQP', constraints=contraintes,bounds=bounds)
    resultat = minimize(objectif, x0, args=(pop,Optimisation[num_school]), method='Nelder-Mead')
    new_USP.append(resultat.x)
    point=Point(resultat.x)
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(5000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    Integ = gpd.sjoin(pop, gpd_buffer, predicate="within", how="inner")
    nombre_patients.append(Integ['population'].sum(axis=0))
    pop=pop.drop(Integ.index)


cmap_opt=plt.cm.get_cmap('hsv',Tot_num_USP)
vmin, vmax = pop_dif_acces["population"].min(), pop_dif_acces["population"].max()#
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15, 15))
pop_dif_acces.plot(ax=ax[0], cmap=cmap, norm=norm,column="population",alpha=0.7)
region.plot(ax=ax[0],facecolor="None",edgecolor="black")
restricted_commune.plot(ax=ax[0],facecolor="yellow",edgecolor="red",alpha=0.5)
for key in Optimisation.keys():
    ax[0].plot(np.array(Optimisation[key]['track'])[:,0],np.array(Optimisation[key]['track'])[:,1],color=cmap_opt(key),label=key)
    point=Point(Optimisation[key]['track'][-1])
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(5000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    gpd_buffer.plot(ax=ax[0],facecolor='green',alpha=0.3)

for key in Optimisation.keys():
    ax[1].plot(np.arange(len(Optimisation[key]['goal'])),Optimisation[key]['goal'],color=cmap_opt(key),label=key)

ax[1].set_xlabel("Itération")
ax[1].set_ylabel("Fonction coût: -1*Population impactée")
plt.show()

resultat_optim_sans_contraintes= pd.DataFrame(columns=["region_nom","prefecture_nom","commune_nom","canton_nom", "nombre de personne couvertes","longitude","latitude"])
for key in Optimisation.keys():
    point=Point(Optimisation[key]['track'][-1])
    gpd_point=gpd.GeoDataFrame([{'geometry':point}],crs="EPSG:32631")
    gpd_loc = gpd.sjoin(gpd_point, canton, predicate="within", how="inner")
    gpd_point=gpd_point.to_crs(4326)
    resultat_optim_sans_contraintes.loc[key]=[gpd_loc['region_nom'][0],gpd_loc['prefecture_nom'][0],gpd_loc['commune_nom'][0],gpd_loc['canton_nom'][0],nombre_patients[key],gpd_point.iloc[0].geometry.x,gpd_point.iloc[0].geometry.y]

resultat_optim_sans_contraintes.to_csv("resultat_optim_sans_contraintes.csv", index=False, encoding='utf-8-sig')
##############################################################################################################
# Optimisation avec restriction de la zone d'implantation
##############################################################################################################
Liste_commune=['Dankpen 1','Dankpen 2','Dankpen 3','Anie 1','Haho 2','Tchamba 1','Tchamba 3','Oti-Sud 1','Oti-Sud 2','Ogou 4']
restricted_commune=commune[commune['commune_nom'].isin(Liste_commune)]
restricted_commune=gpd.GeoDataFrame({"geometry":[restricted_commune['geometry'].union_all()]},crs=commune.crs)
def evaluate_good_spot(updated_pop,Target_point):
    Find_good_spot=Target_point.copy()
    Find_good_spot['buffer']=Find_good_spot.geometry.buffer(5000)
    Find_good_spot['geometry'] = Find_good_spot['buffer']
    Find_good_spot = Find_good_spot.set_geometry('buffer')
    pop_cumul=updated_pop.copy()
    pop_cumul['centroid'] = pop_cumul.geometry.centroid
    centroids = gpd.GeoDataFrame(pop_cumul[['population']], geometry=pop_cumul['centroid'], crs=updated_pop.crs)
    joined = gpd.sjoin(centroids, Find_good_spot, predicate='within', how='inner')
    agg_pop_large_dense = joined.groupby('index_right').agg(
        Pop=('population', 'sum'))
    agg_pop_large_dense.reset_index(inplace=True)
    agg_pop_large_dense.rename(columns={'index_right': 'index'}, inplace=True)   
    Find_good_spot.reset_index(inplace=True) 
    agg_pop_large_dense = agg_pop_large_dense.merge(Find_good_spot, on='index', how='inner')
    agg_pop_large_dense=agg_pop_large_dense[['Pop','buffer']]
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('buffer')
    agg_pop_large_dense['centroid']=agg_pop_large_dense.geometry.centroid
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('centroid')
    return agg_pop_large_dense


new_USP=[]
nombre_patients=[]
Optimisation={}
Tot_num_USP=20
pop=pop_dif_acces.copy()
for num_school in range(Tot_num_USP):
    Optimisation[num_school]={}
    Optimisation[num_school]['track']=[]
    Optimisation[num_school]['goal']=[]
    def objectif(x, pop,Optimisation_dict):
        population_a_optim = pop.copy()
        Optimisation_dict['track'].append(x)
        point = Point(x[0], x[1])
        gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
        gpd_buffer=gpd_buffer.buffer(5000)
        gpd_buffer.set_crs("EPSG:32631", inplace=True)
        gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
        Integ = gpd.sjoin(population_a_optim, gpd_buffer, predicate="within", how="inner")  
        Tot_integ = -Integ['population'].sum(axis=0)
        Optimisation_dict['goal'].append(Tot_integ)
        return Tot_integ
    # Contrainte sur la restriction du point qui doit être à l'intérieur de la zone
    def geo_constraint(x, restricted_commune):
        point = Point(x[0], x[1])
        is_inside = restricted_commune.contains(point).any()
        return 1 if is_inside else -1
    # Définir la contrainte comme un dictionnaire
    constraints = [{
    'type': 'ineq',  # pour une contrainte du type f(x) >= 0
    'fun': lambda x: geo_constraint(x, restricted_commune)
    }]    
    bounds = [(0, 100*201200), (680000, 1248000)]
    # Point de départ
    pop_dif_acces_point=gpd.GeoDataFrame({"geometry":pop.geometry.centroid},crs=pop.crs)
    Target_point=gpd.sjoin(pop_dif_acces_point,commune[commune['commune_nom'].isin(Liste_commune)], predicate='within', how='inner')
    Target_point=Target_point[['geometry']]
    agg_pop_large_dense=evaluate_good_spot(pop,Target_point)
    Ligne=agg_pop_large_dense[agg_pop_large_dense['Pop']==agg_pop_large_dense['Pop'].max()].centroid
    x0 = [Ligne.geometry.iloc[0].x, Ligne.geometry.iloc[0].y]
    agg_pop_large_dense.drop(Ligne.index,inplace=True)
    resultat = minimize(objectif, x0, args=(pop,Optimisation[num_school]), method='SLSQP',constraints=constraints,options={'disp': True, 'maxiter': 100, 'eps': 5000})
    new_USP.append(resultat.x)
    point=Point(resultat.x)
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(5000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    Integ = gpd.sjoin(pop, gpd_buffer, predicate="within", how="inner")
    nombre_patients.append(Integ['population'].sum(axis=0))
    pop=pop.drop(Integ.index)




cmap_opt=plt.cm.get_cmap('hsv',Tot_num_USP)
vmin, vmax = pop_dif_acces["population"].min(), pop_dif_acces["population"].max()#
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15, 15))
pop_dif_acces.plot(ax=ax[0], cmap=cmap, norm=norm,column="population",alpha=0.7)
region.plot(ax=ax[0],facecolor="None",edgecolor="black")
restricted_commune.plot(ax=ax[0],facecolor="yellow",edgecolor="red",alpha=0.5)
for key in Optimisation.keys():
    ax[0].plot(np.array(Optimisation[key]['track'])[:,0],np.array(Optimisation[key]['track'])[:,1],color=cmap_opt(key),label=key)
    point=Point(Optimisation[key]['track'][-1])
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(5000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    gpd_buffer.plot(ax=ax[0],facecolor='green',alpha=0.3)

for key in Optimisation.keys():
    ax[1].plot(np.arange(len(Optimisation[key]['goal'])),Optimisation[key]['goal'],color=cmap_opt(key),label=key)

ax[1].set_xlabel("Itération")
ax[1].set_ylabel("Fonction coût: -1*Population impactée")
plt.show()



resultat_optim_avec_contraintes= pd.DataFrame(columns=["region_nom","prefecture_nom","commune_nom","canton_nom", "nombre de personne couvertes","longitude","latitude"])
for key in Optimisation.keys():
    point=Point(Optimisation[key]['track'][-1])
    gpd_point=gpd.GeoDataFrame([{'geometry':point}],crs="EPSG:32631")
    gpd_loc = gpd.sjoin(gpd_point, canton, predicate="within", how="inner")
    gpd_point=gpd_point.to_crs(4326)
    resultat_optim_avec_contraintes.loc[key]=[gpd_loc['region_nom'][0],gpd_loc['prefecture_nom'][0],gpd_loc['commune_nom'][0],gpd_loc['canton_nom'][0],nombre_patients[key],gpd_point.iloc[0].geometry.x,gpd_point.iloc[0].geometry.y]

resultat_optim_avec_contraintes.to_csv("resultat_optim_avec_contraintes.csv", index=False, encoding='utf-8-sig')


raise SystemExit



Fontsize_title=22
Fontsize_label=19
Fontsize_tick_cbar=17

vmin, vmax = Pop_dif_prefecture["percent/pop_pref"].min(), Pop_dif_prefecture["percent/pop_pref"].max()  #
vmin, vmax = cmap_bounds[0], cmap_bounds[1]
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(figsize=(15, 15))
Pop_dif_prefecture.plot(ax=ax, cmap=cmap, norm=norm,column="percent/pop_pref",alpha=0.7)
region.plot(ax=ax,facecolor="None",edgecolor="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(0.67, 0.98, "Population en difficulté d'accès\n(en pourcentage de la pop. prefectorale)",  # texte
        transform=ax.transAxes,         # coordonnées relatives à l'axe (0–1)
        ha='center', va='bottom',       # alignement horizontal / vertical
        fontsize=Fontsize_title)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Obligatoire pour éviter une erreur
cax = inset_axes(ax,
                width="5%",       # % de la largeur de l'axe parent
                height="50%",     # % de la hauteur de l'axe parent
                loc='lower left',
                bbox_to_anchor=(1.0, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0)
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label("% de la population préfectorale",fontsize=Fontsize_label)
cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
rect = Rectangle(
    (0.36, 0.115), 0.4, 0.81,                        # position et taille (x, y, width, height)
    linewidth=2,                        # épaisseur du trait
    edgecolor='black',                  # couleur de bord
    facecolor='none',                   # transparent à l'intérieur
    transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
    zorder=1000                         # bien au-dessus de tout le reste
)
# fig.patches.append(rect)
plt.show()

























########################################################################################################
# Internat
def evaluate_good_spot(updated_pop):
    Find_good_spot=updated_pop.copy()
    Find_good_spot['buffer']=Find_good_spot.geometry.buffer(9000)
    Find_good_spot['geometry'] = Find_good_spot['buffer']
    Find_good_spot = Find_good_spot.set_geometry('buffer')
    pop_cumul=updated_pop.copy()
    pop_cumul['centroid'] = pop_cumul.geometry.centroid
    centroids = gpd.GeoDataFrame(pop_cumul[['5_11']], geometry=pop_cumul['centroid'], crs=updated_pop.crs)
    joined = gpd.sjoin(centroids, Find_good_spot, predicate='within', how='inner')
    agg_pop_large_dense = joined.groupby('index_right').agg(
        Pop=('5_11_left', 'sum'))
    agg_pop_large_dense.reset_index(inplace=True)
    agg_pop_large_dense.rename(columns={'index_right': 'index'}, inplace=True)   
    Find_good_spot.reset_index(inplace=True) 
    agg_pop_large_dense = agg_pop_large_dense.merge(Find_good_spot, on='index', how='inner')
    agg_pop_large_dense=agg_pop_large_dense[['Pop','buffer']]
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('buffer')
    agg_pop_large_dense['centroid']=agg_pop_large_dense.geometry.centroid
    agg_pop_large_dense=agg_pop_large_dense.set_geometry('centroid')
    return agg_pop_large_dense


new_school=[]
nombre_eleve=[]
Optimisation={}
Tot_num_school=10
pop=pop_dif_acces.copy()
for num_school in range(Tot_num_school):
    Optimisation[num_school]={}
    Optimisation[num_school]['track']=[]
    Optimisation[num_school]['goal']=[]
    def objectif(x, pop,Optimisation_dict):
        population_a_optim = pop.copy()
        Optimisation_dict['track'].append(x)
        point = Point(x[0], x[1])
        gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
        gpd_buffer=gpd_buffer.buffer(9000)
        gpd_buffer.set_crs("EPSG:32631", inplace=True)
        gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
        Integ = gpd.sjoin(population_a_optim, gpd_buffer, predicate="within", how="inner")  
        Tot_integ = -Integ['5_11'].sum(axis=0)
        Optimisation_dict['goal'].append(Tot_integ)
        return Tot_integ
    bounds = [(0, 201200), (680000, 1248000)]
    # Point de départ
    agg_pop_large_dense=evaluate_good_spot(pop)
    Ligne=agg_pop_large_dense[agg_pop_large_dense['Pop']==agg_pop_large_dense['Pop'].max()].centroid
    x0 = [Ligne.geometry.iloc[0].x, Ligne.geometry.iloc[0].y]
    agg_pop_large_dense.drop(Ligne.index,inplace=True)
    # Appel à l’optimiseur
    # resultat = minimize(objectif, x0, args=(pop), method='SLSQP', constraints=contraintes,bounds=bounds)
    resultat = minimize(objectif, x0, args=(pop,Optimisation[num_school]), method='Nelder-Mead')
    new_school.append(resultat.x)
    point=Point(resultat.x)
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(9000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    Integ = gpd.sjoin(pop, gpd_buffer, predicate="within", how="inner")
    nombre_eleve.append(Integ['5_11'].sum(axis=0))
    pop=pop.drop(Integ.index)


cmap_opt=plt.cm.get_cmap('hsv',Tot_num_school)
vmin, vmax = population_outside["5_11"].min(), population_outside["5_11"].max()#
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15, 15))
population_outside.plot(ax=ax[0], cmap=cmap, norm=norm,column="5_11",alpha=0.7)
region.plot(ax=ax[0],facecolor="None",edgecolor="black")
for key in Optimisation.keys():
    ax[0].plot(np.array(Optimisation[key]['track'])[:,0],np.array(Optimisation[key]['track'])[:,1],color=cmap_opt(key),label=key)
    point=Point(Optimisation[key]['track'][-1])
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(9000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    gpd_buffer.plot(ax=ax[0],facecolor='green',alpha=0.3)

ctx.add_basemap(ax=ax[0], crs=agg_pop_outside_canton.crs,attribution=False)

for key in Optimisation.keys():
    ax[1].plot(np.arange(len(Optimisation[key]['goal'])),Optimisation[key]['goal'],color=cmap_opt(key),label=key)

plt.show()