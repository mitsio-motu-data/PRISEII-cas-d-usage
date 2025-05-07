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

############################################################################################################
pop_RGPH=gpd.read_file("./Pop_Togo/Pop_density_map_Meta/grille_200m_togo.gpkg")
Ecole_primaire=gpd.read_file("./Donnees_geoportail/educ_etablissements_scolaires.gpkg")

# Map
commune=gpd.read_file("./map/14_11_22_Communes_du_Togo_2.gpkg")
canton=gpd.read_file("./map/14_11_22_Togo_Cantons2_4326.gpkg")
prefecture=gpd.read_file("./map/15_01_25_Togo_Prefecture_4326.gpkg")
region=gpd.read_file("./map/regions_Togo.gpkg")
Lome=gpd.read_file("./map/2024-05-31_Grand_Lomé.gpkg")

# Hexagones_classifies
hexagones_classifies=gpd.read_file("./map/hexagones_classifies.gpkg")
hexagones_classifies=hexagones_classifies[['final_class','geometry']]
# Conversion en mètre
pop_RGPH=pop_RGPH.to_crs(epsg=32631)
Ecole_primaire=Ecole_primaire.to_crs(epsg=32631)
commune=commune.to_crs(epsg=32631)
canton=canton.to_crs(epsg=32631)
prefecture=prefecture.to_crs(epsg=32631)
region=region.to_crs(epsg=32631)
hexagones_classifies=hexagones_classifies.to_crs(epsg=32631)
Lome=Lome.to_crs(epsg=32631)

### Pref without agoè-Nyivé & Golfe
pref_cp=prefecture.copy()
pref_cp=pref_cp[~pref_cp['prefecture_nom'].isin(['Agoè-Nyivé', 'Golfe'])]
pref_cp = pref_cp.dissolve(by='region_nom', as_index=False)
region_sans_lome=pref_cp[['region_nom','geometry']]
Lome['region_nom']='Lome'
region_avec_lome=gdf_concat = pd.concat([region_sans_lome, Lome[['region_nom','geometry']]], ignore_index=True)
region_avec_lome=gpd.GeoDataFrame(region_avec_lome,geometry='geometry',crs=region.crs)

##
gpd_pop=gpd.GeoDataFrame(pop_RGPH[['total_pop_meta','geometry']],geometry=pop_RGPH['geometry'],crs=pop_RGPH.crs)
gpd_pop['total_pop_meta']=(8095000/pop_RGPH['total_pop_meta'].sum())*pop_RGPH['total_pop_meta']
gpd_pop['centroid']=gpd_pop.geometry.centroid
gpd_pop=gpd_pop.set_geometry('centroid')

############################################################################################################
# gpd_pop = gpd.GeoDataFrame(pop, geometry=gpd.points_from_xy(pop.longitude, pop.latitude), crs="EPSG:4326")
gpd_pop=gpd_pop.rename(columns={'total_pop_meta':'population'})
gpd_pop=gpd.GeoDataFrame(gpd_pop[['population','geometry']],geometry=gpd_pop['geometry'],crs=gpd_pop.crs)
gpd_pop['pop_percent']=gpd_pop['population']/(gpd_pop['population'].sum(axis=0))
#
gpd_pop['5_11']=1190346*gpd_pop['pop_percent']+1/4*910696*gpd_pop['pop_percent']
gpd_pop=gpd_pop[gpd_pop['5_11']>0.001]

########################################################################################################

Ecole_primaire_fil=Ecole_primaire[(Ecole_primaire['etablissement_categorie']=="Ecole primaire") & \
                                  (Ecole_primaire['etablissement_secteur']=="Public")]

from sklearn.neighbors import BallTree
import time
start_time = time.time()
coords_primaire = np.array(list(zip(Ecole_primaire_fil.geometry.x, Ecole_primaire_fil.geometry.y)))
tree = BallTree(coords_primaire, leaf_size=200, metric='euclidean')
# Points à tester
centroids = gpd_pop.geometry.centroid
coords_pop = np.array(list(zip(centroids.x, centroids.y)))
# Trouver les plus proches voisins
distances, indices = tree.query(coords_pop, k=1)
# Remplir la colonne
gpd_pop['distance_ecole_proche'] = distances
end_time= time.time()
print(f"Elapsed time computing minimal distance : {end_time-start_time} secondes")
   
Classified_pop=gpd.sjoin(gpd_pop, hexagones_classifies, predicate="within", how="left")
Classified_pop.loc[Classified_pop['final_class'].isna(),'final_class']='rural'
population_outside_1=Classified_pop[(Classified_pop['distance_ecole_proche']>2000) & (Classified_pop['final_class']=='urbain')]
population_outside_2=Classified_pop[(Classified_pop['distance_ecole_proche']>3000) & (Classified_pop['final_class']=='rural')]
population_outside = pd.concat([population_outside_1, population_outside_2], axis=0)
print(f" Population outise : {population_outside['5_11'].sum(axis=0)}")
population_outside.drop(columns='index_right',inplace=True)

distance_pondere=(gpd_pop['distance_ecole_proche'] * gpd_pop['5_11']).sum(axis=0)/(gpd_pop['5_11'].sum(axis=0))
print(f"Population à plus de 2 km : {gpd_pop[gpd_pop['distance_ecole_proche']>2000]['5_11'].sum(axis=0)}")
print(f"Distance moyenne: {distance_pondere}")

#########################################################################################################
#########################################################################################################     
#  Distance
#########################################################################################################
#########################################################################################################
def Agg_zone(gdf,zone,zone_name):
    if 'centroid' not in gdf.columns:  
        gdf['centroid'] = gdf['geometry'].centroid
    gdf=gdf.set_geometry('centroid')
    gdf=gdf.reset_index()
    pop_par_zone = gpd.sjoin(gdf, zone, predicate='within', how='left')
    pop_par_zone['weighted_distance'] = pop_par_zone['population'] * pop_par_zone['distance_ecole_proche']
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
    if screen=='laptop':
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
        ax.text(0.67, 0.98, "Distance moyenne des élèves\naux écoles primaires",  # texte
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
    else:
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
        ax.text(0.67, 0.98, "Distance moyenne des élèves\naux écoles primaires",  # texte
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
print_agg_distance_over_zone(mean_dist_communes,'pc_fixe')
print_agg_distance_over_zone(mean_dist_canton,'pc_fixe')
print_agg_distance_over_zone(mean_dist_region_avec_lome,'pc_fixe')

mean_dist_region[['region_nom','weighted_avg_distance']].to_csv("distance_moyenne_region.csv", index=False, encoding='utf-8-sig')
mean_dist_prefecture[['region_nom','prefecture_nom','weighted_avg_distance']].to_csv("distance_moyenne_prefecture.csv", index=False, encoding='utf-8-sig')
mean_dist_communes[['region_nom','prefecture_nom','commune_nom','weighted_avg_distance']].to_csv("distance_moyenne_commune.csv", index=False, encoding='utf-8-sig')
mean_dist_canton[['region_nom','prefecture_nom','commune_nom','canton_nom','weighted_avg_distance']].to_csv("distance_moyenne_canton.csv", index=False, encoding='utf-8-sig')
mean_dist_region_avec_lome[['region_nom','weighted_avg_distance']].to_csv("distance_moyenne_region_avec_lome.csv", index=False, encoding='utf-8-sig')


mean_dist_prefecture=mean_dist_prefecture.sort_values(by='weighted_avg_distance', ascending=False)
#########################################################################################################     
#  Population exclu
#########################################################################################################
#########################################################################################################
def Agg_zone_exclu(gdf,zone,zone_name):
    if 'centroid' not in gdf.columns:  
        gdf['centroid'] = gdf['geometry'].centroid
    gdf=gdf.set_geometry('centroid')
    gdf=gdf.reset_index()
    pop_par_zone = gpd.sjoin(gdf, zone, predicate='within', how='left')
    agg_zone = pop_par_zone.groupby(zone_name).agg(
        population_exclu=('5_11', 'sum'),
        total_population=('population', 'sum'),
        zone_name=(zone_name, 'first')
    )
    agg_zone = agg_zone.merge(zone, on=zone_name, how='right')
    agg_zone=gpd.GeoDataFrame(agg_zone,geometry='geometry',crs=gdf.crs)
    agg_zone['population_exclu'] = agg_zone['population_exclu'].fillna(0)
    return agg_zone

def print_pop_exclu_over_zone(mean_agg,screen,cmap_bounds="None"):
    if screen=='laptop':
        Fontsize_title=14
        Fontsize_label=12
        Fontsize_tick_cbar=10
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["population_exclu"].min(), mean_agg["population_exclu"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="population_exclu",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Nombres d'élèves en difficulté d'accès",  # texte
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
        cbar.set_label("Nombre d'élèves",fontsize=Fontsize_label)
        cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
        rect = Rectangle(
            (0.41, 0.115), 0.26, 0.81,                        # position et taille (x, y, width, height)
            linewidth=2,                                      # épaisseur du trait
            edgecolor='black',                                # couleur de bord
            facecolor='none',                                 # transparent à l'intérieur
            transform=fig.transFigure,                        # important : coordonnées en fig (et non ax)
            zorder=1000                                       # bien au-dessus de tout le reste
        )
        fig.patches.append(rect)
        plt.show()
    else:
        Fontsize_title=20
        Fontsize_label=18
        Fontsize_tick_cbar=16
        if cmap_bounds=="None":
            vmin, vmax = mean_agg["population_exclu"].min(), mean_agg["population_exclu"].max()  #
        else :
            vmin, vmax = cmap_bounds[0], cmap_bounds[1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        fig, ax = plt.subplots(figsize=(15, 15))
        mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column="population_exclu",alpha=0.7)
        region.plot(ax=ax,facecolor="None",edgecolor="black")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.67, 0.98, "Nombres d'élèves en difficulté d'accès",  # texte
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
        cbar.set_label("Nombre d'élèves",fontsize=Fontsize_label)
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

pop_outside_per_region=Agg_zone_exclu(population_outside.copy(),region,'region_nom')
pop_outside_per_prefecture=Agg_zone_exclu(population_outside.copy(),prefecture,'prefecture_nom')
pop_outside_per_commune=Agg_zone_exclu(population_outside.copy(),commune,'commune_nom')
pop_outside_per_canton=Agg_zone_exclu(population_outside.copy(),canton,'canton_nom')
pop_outside_per_region_avec_lome=Agg_zone_exclu(population_outside.copy(),region_avec_lome,'region_nom')

print_pop_exclu_over_zone(pop_outside_per_region,'pc_fixe',cmap_bounds="None")
print_pop_exclu_over_zone(pop_outside_per_prefecture,'pc_fixe',cmap_bounds="None")
print_pop_exclu_over_zone(pop_outside_per_commune,'pc_fixe',cmap_bounds="None")
print_pop_exclu_over_zone(pop_outside_per_canton,'pc_fixe',cmap_bounds="None")

pop_outside_per_region[['region_nom','population_exclu']].to_csv("population_exclu_region.csv", index=False, encoding='utf-8-sig')
pop_outside_per_prefecture[['region_nom','prefecture_nom','population_exclu']].to_csv("population_exclu_prefecture.csv", index=False, encoding='utf-8-sig')
pop_outside_per_commune[['region_nom','prefecture_nom','commune_nom','population_exclu']].to_csv("population_exclu_commune.csv", index=False, encoding='utf-8-sig')
pop_outside_per_canton[['region_nom','prefecture_nom','commune_nom','canton_nom','population_exclu']].to_csv("population_exclu_canton.csv", index=False, encoding='utf-8-sig')
pop_outside_per_region_avec_lome[['region_nom','population_exclu']].to_csv("population_exclu_region_avec_lome.csv", index=False, encoding='utf-8-sig')
######################################################################
# Ecole rurale
def evaluate_good_spot(updated_pop):
    Find_good_spot=updated_pop.copy()
    Find_good_spot['buffer']=Find_good_spot.geometry.buffer(3000)
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
Tot_num_school=5
population_outside['centroid']=population_outside.geometry.centroid
population_outside=population_outside.set_geometry('centroid')
pop=population_outside.copy()

for num_school in range(Tot_num_school):
    Optimisation[num_school]={}
    Optimisation[num_school]['track']=[]
    Optimisation[num_school]['goal']=[]
    def objectif(x, pop,Optimisation_dict):
        population_a_optim = pop.copy()
        Optimisation_dict['track'].append(x)
        point = Point(x[0], x[1])
        gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
        gpd_buffer=gpd_buffer.buffer(3000)
        gpd_buffer.set_crs("EPSG:32631", inplace=True)
        gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
        Integ = gpd.sjoin(population_a_optim, gpd_buffer, predicate="within", how="inner")  
        Tot_integ = -Integ['5_11'].sum(axis=0)
        Optimisation_dict['goal'].append(Tot_integ)
        return Tot_integ
    bounds = [(0, 501200), (680000, 1248000)]
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
    gpd_buffer=gpd_buffer.buffer(3000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    Integ = gpd.sjoin(pop, gpd_buffer, predicate="within", how="inner")
    nombre_eleve.append(Integ['5_11'].sum(axis=0))
    pop=pop.drop(Integ.index)

population_outside=population_outside.set_geometry('geometry')
cmap_opt=plt.cm.get_cmap('hsv',Tot_num_school)
vmin, vmax = population_outside["5_11"].min(), population_outside["5_11"].max()#
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15, 10))
population_outside.plot(ax=ax[0], cmap=cmap, norm=norm,column="5_11",alpha=0.7)
region.plot(ax=ax[0],facecolor="None",edgecolor="black")
for key in Optimisation.keys():
    ax[0].plot(np.array(Optimisation[key]['track'])[:,0],np.array(Optimisation[key]['track'])[:,1],color=cmap_opt(key),label=key)
    point=Point(Optimisation[key]['track'][-1])
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(5000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    gpd_buffer.plot(ax=ax[0],facecolor='green',alpha=0.7)

ax[0].set_title("Localisation des nouvelles écoles et trajectoire de l'optim")
ctx.add_basemap(ax=ax[0], crs=population_outside.crs,attribution=False)
for key in Optimisation.keys():
    ax[1].plot(np.arange(len(Optimisation[key]['goal'])),Optimisation[key]['goal'],color=cmap_opt(key),label=key)

ax[1].set_xlabel("Itération")
ax[1].set_ylabel('Fonction cout (-1*Nb élèves)')
ax[1].set_title("Performance de l'optimisation pour chaque école")
plt.show()



resultat_optim_avec_contraintes= pd.DataFrame(columns=["region_nom","prefecture_nom","commune_nom","canton_nom", "nombre_eleve","longitude","latitude"])
for key in Optimisation.keys():
    point=Point(Optimisation[key]['track'][-1])
    gpd_point=gpd.GeoDataFrame([{'geometry':point}],crs="EPSG:32631")
    gpd_loc = gpd.sjoin(gpd_point, canton, predicate="within", how="inner")
    gpd_point=gpd_point.to_crs(4326)
    resultat_optim_avec_contraintes.loc[key]=[gpd_loc['region_nom'][0],gpd_loc['prefecture_nom'][0],gpd_loc['commune_nom'][0],gpd_loc['canton_nom'][0],nombre_eleve[key],gpd_point.iloc[0].geometry.x,gpd_point.iloc[0].geometry.y]

resultat_optim_avec_contraintes=resultat_optim_avec_contraintes.sort_values(by='nombre_eleve', ascending=False)
resultat_optim_avec_contraintes.to_csv("resultat_optim_avec_contraintes.csv", index=False, encoding='utf-8-sig')



Ecole_primaire_fil=Ecole_primaire_fil.to_crs(4326)
Export_Ecole_primaire=Ecole_primaire_fil.copy()
Export_Ecole_primaire=Export_Ecole_primaire[['region','prefecture','commune','canton','etablissement_nom','geometry']]
Export_Ecole_primaire["longitude"] = Export_Ecole_primaire.geometry.x
Export_Ecole_primaire["latitude"] = Export_Ecole_primaire.geometry.y
Export_Ecole_primaire.drop(columns='geometry').to_csv("liste_Ecole_primaire_publiques.csv", index=False, encoding='utf-8-sig')

raise SystemExit











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
pop=population_outside.copy()
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
    resultat = minimize(objectif, x0, args=(pop,Optimisation[num_school]), method='Nelder-Mead',bounds=bounds)
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
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15, 10))
population_outside.plot(ax=ax[0], cmap=cmap, norm=norm,column="5_11",alpha=0.7)
region.plot(ax=ax[0],facecolor="None",edgecolor="black")
for key in Optimisation.keys():
    ax[0].plot(np.array(Optimisation[key]['track'])[:,0],np.array(Optimisation[key]['track'])[:,1],color=cmap_opt(key),label=key)
    point=Point(Optimisation[key]['track'][-1])
    gpd_buffer=gpd.GeoDataFrame([{'geometry':point}])
    gpd_buffer=gpd_buffer.buffer(9000)
    gpd_buffer.set_crs("EPSG:32631", inplace=True)
    gpd_buffer = gpd.GeoDataFrame(geometry=gpd_buffer)
    gpd_buffer.plot(ax=ax[0],facecolor='green',alpha=0.7)

ax[0].set_title("Localisation des nouvelles écoles et trajectoire de l'optim")
ctx.add_basemap(ax=ax[0], crs=agg_pop_outside_canton.crs,attribution=False)
for key in Optimisation.keys():
    ax[1].plot(np.arange(len(Optimisation[key]['goal'])),Optimisation[key]['goal'],color=cmap_opt(key),label=key)

ax[1].set_xlabel("Itération")
ax[1].set_ylabel('Fonction cout (-1*Nb élèves)')
ax[1].set_title("Performance de l'optimisation pour chaque école")
plt.show()