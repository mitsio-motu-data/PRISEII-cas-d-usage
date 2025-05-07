import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.optimize import minimize
import numpy as np
import contextily as ctx

############################################################################################################
pop=gpd.read_file("./Pop_Togo/Pop_density_map_Meta/cantons_population_2.gpkg")
population_totale_du_togo=pop['pop_rgph_2020'].sum(axis=0)
pop_commune=pop.copy().groupby(['commune_nom']).agg(Population =('pop_rgph_2020','sum'))
pop_prefecture=pop.copy().groupby(['prefecture_nom']).agg(Population =('pop_rgph_2020','sum'))
pop_region=pop.copy().groupby(['region_nom']).agg(Population =('pop_rgph_2020','sum'))

pop_hex=gpd.read_file("./Pop_Togo/Pop_density_map_Meta/grille_200m_togo.gpkg")
# Dataset sante
Centres_sante=gpd.read_file("./Donnees_geoportail/sant_etablissements_sante.gpkg")
Admin_sante=gpd.read_file("./Donnees_geoportail/admin_sante.gpkg")
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
pop=pop.to_crs(epsg=32631)
Centres_sante=Centres_sante.to_crs(epsg=32631)
liste_public=['Communautaire', 'ONG', 'Confessionnel', 'Public',
       'Association', 'ONG /Associatif', 'Public/Communautaire',
       'Confessionnel/communautaire', 'Para Public']
Centres_sante_publics=Centres_sante[Centres_sante['secteur'].isin(liste_public)]
Centres_sante_prive=Centres_sante[~Centres_sante['secteur'].isin(liste_public)]

Liste_retenue=['Hopital 1','Hopital 2','CMS','CHU','CHR','Hopital Spécialisé']
Centre_lits=Centres_sante.copy()
Centre_lits=Centre_lits[Centre_lits['etablissement_type'].isin(Liste_retenue)].copy()
Centre_lits=Centre_lits[['region','prefecture','commune','canton','nom_fs',\
    'etablissement_type','secteur','services_proposes','service_hospitalisation','lits_total_nbr','lits_bon_nbr',\
        'lits_maternite_nbr','geometry']]


###################################################################################################################################################
###################################################################################################################################################
### On ajoute à la mais les centres Public pour lesquels le nobmre de lits est nul
# A partir du Annuaire-2019_final_V2.pdf
#  
Centre_lits.loc[Centre_lits['nom_fs']=="CHR Lome Commune",'lits_total_nbr']=139
Centre_lits.loc[Centre_lits['nom_fs']=="CHR Lome Commune",'lits_bon_nbr']=139
Centre_lits.loc[Centre_lits['nom_fs']=="CHR Lome Commune",'lits_maternite_nbr']=22

Centre_lits.loc[Centre_lits['nom_fs']=="CHP Sotouboua",'lits_total_nbr']=89
Centre_lits.loc[Centre_lits['nom_fs']=="CHP Sotouboua",'lits_bon_nbr']=89
Centre_lits.loc[Centre_lits['nom_fs']=="CHP Sotouboua",'lits_maternite_nbr']=13

Centre_lits.loc[Centre_lits['nom_fs']=="Polyclinique Internationale Saint Joseph",'lits_total_nbr']=72
Centre_lits.loc[Centre_lits['nom_fs']=="Polyclinique Internationale Saint Joseph",'lits_bon_nbr']=72
Centre_lits.loc[Centre_lits['nom_fs']=="Polyclinique Internationale Saint Joseph",'lits_maternite_nbr']=17

Centre_lits.loc[Centre_lits['nom_fs']=="CHP Notse",'lits_total_nbr']=136
Centre_lits.loc[Centre_lits['nom_fs']=="CHP Notse",'lits_bon_nbr']=136
Centre_lits.loc[Centre_lits['nom_fs']=="CHP Notse",'lits_maternite_nbr']=28

###################################################################################################################################################
###################################################################################################################################################


# Centre_lits.drop(columns='geometry').to_excel("Centre_lits.xlsx", index=False)
Centre_lits.loc[Centre_lits['lits_total_nbr'].isna(),'lits_total_nbr']=0

Centres_sante_publics.drop(columns='geometry').to_excel("Centres_sante_publics.xlsx", index=False)
Centres_sante_prive.drop(columns='geometry').to_excel("Centres_sante_prive.xlsx", index=False)
Admin_sante.drop(columns='geometry').to_excel("Admin_sante.xlsx", index=False)

Centre_lits.loc[Centre_lits['lits_total_nbr'] < Centre_lits['lits_bon_nbr'], 'lits_total_nbr'] = \
    Centre_lits.loc[Centre_lits['lits_total_nbr'] < Centre_lits['lits_bon_nbr'], 'lits_bon_nbr']

Centre_lits.loc[Centre_lits['lits_total_nbr']<Centre_lits['lits_maternite_nbr'],'lits_total_nbr']=\
    Centre_lits.loc[Centre_lits['lits_total_nbr']<Centre_lits['lits_maternite_nbr'],'lits_total_nbr']+\
        Centre_lits.loc[Centre_lits['lits_total_nbr'] < Centre_lits['lits_maternite_nbr'], 'lits_maternite_nbr']

Centre_lits.loc[Centre_lits['lits_bon_nbr']<Centre_lits['lits_maternite_nbr'],'lits_bon_nbr']=\
    Centre_lits.loc[Centre_lits['lits_bon_nbr']<Centre_lits['lits_maternite_nbr'],'lits_bon_nbr']+\
        Centre_lits.loc[Centre_lits['lits_total_nbr'] < Centre_lits['lits_maternite_nbr'], 'lits_maternite_nbr']     

Centre_lits.loc[(Centre_lits['lits_bon_nbr'].isna()) & (Centre_lits['lits_total_nbr'].notna()),'lits_bon_nbr']=\
    Centre_lits.loc[(Centre_lits['lits_bon_nbr'].isna()) & (Centre_lits['lits_total_nbr'].notna()),'lits_total_nbr']   

Centre_lits.loc[(Centre_lits['lits_bon_nbr']==0) & (Centre_lits['lits_total_nbr']!=0),'lits_bon_nbr']=\
    Centre_lits.loc[(Centre_lits['lits_bon_nbr']==0) & (Centre_lits['lits_total_nbr']!=0),'lits_total_nbr']   


Centre_lits.loc[(Centre_lits['lits_total_nbr']!=0) & (Centre_lits['service_hospitalisation']=='Non'),'service_hospitalisation'] = 'Oui'
Centre_lits.drop(columns='geometry').to_excel("Bilan_lits_ZIGAN.xlsx", index=False)



liste_public=['Communautaire', 'ONG', 'Confessionnel', 'Public',
       'Association', 'ONG /Associatif', 'Public/Communautaire',
       'Confessionnel/communautaire', 'Para Public']
Centre_lits.loc[Centre_lits['secteur'].isin(liste_public),'secteur']='Public'
Centre_lits.loc[~Centre_lits['secteur'].isin(liste_public),'secteur']='Prive'
Centre_hospitalisation=Centre_lits[Centre_lits['service_hospitalisation']=='Oui'].copy()
Centre_hospitalisation.drop(columns='geometry').to_excel("Centre_hospitalisation.xlsx", index=False)
Centre_hospitalisation_nonnul=Centre_hospitalisation[Centre_hospitalisation['lits_total_nbr']!=0]

Admin_sante=Admin_sante.to_crs(epsg=32631)
commune=commune.to_crs(epsg=32631)
canton=canton.to_crs(epsg=32631)
prefecture=prefecture.to_crs(epsg=32631)
region=region.to_crs(epsg=32631)
hexagones_classifies=hexagones_classifies.to_crs(epsg=32631)
Lome=Lome.to_crs(epsg=32631)
pop_hex=pop_hex.to_crs(epsg=32631)
grille_hex=pop_hex.copy()[['geometry']]
grille_hex=grille_hex.to_crs(epsg=32631)
pop_hex=pop_hex[pop_hex['total_pop_meta']>0.001]

prefecture_tot=pd.concat([prefecture[['prefecture_nom','region_nom','geometry']], Lome[['prefecture_nom','region_nom','geometry']]], axis=0)
prefecture_tot=prefecture_tot.reset_index(drop=True)


## On va créer une aggrégation sur la zone géographique : commune,prefecture,region et fonction du secteur (lits_bon_nbr pour le public et lits_bon_nbr pour le privé)
def Aggregation_sectoriel_et_geo(Centre_hospitalisation,pop_zone_admin,Col_admin_zone_name,zone_admin):
    Centre_hospitalisation_commune_secteur = Centre_hospitalisation.groupby([Col_admin_zone_name, 'secteur']).agg(
        lits_total_nbr=('lits_total_nbr', 'sum'),
        lits_bon_nbr=('lits_bon_nbr', 'sum'),
        lits_maternite_nbr=('lits_maternite_nbr', 'sum')
    )
    ## Pour conserver un Dataset avec toutes les zones admin
    secteurs = ['Public', 'Prive']
    # Produit cartésien entre toutes les communes et les deux secteurs
    zone_admin_df = zone_admin[[Col_admin_zone_name+str('_nom')]].drop_duplicates().assign(key=1)
    zone_admin_df.rename(columns={Col_admin_zone_name+str('_nom'):Col_admin_zone_name }, inplace=True)
    secteur_df = pd.DataFrame({'secteur': secteurs}).assign(key=1)
    base_complete = zone_admin_df.merge(secteur_df, on='key').drop('key', axis=1)
    # On merge avec l'aggrégation précédente pour conserver tout les zones
    Centre_hospitalisation_commune_secteur_full = base_complete.merge(
    Centre_hospitalisation_commune_secteur,
    on=[Col_admin_zone_name, 'secteur'],
    how='left'  # pour conserver toutes les combinaisons
    )
    print(Centre_hospitalisation_commune_secteur)
    Centre_hospitalisation_commune_secteur_full.fillna(0, inplace=True)
    # Pivot pour le type Prive, Public
    Centre_hospitalisation_commune_secteur_full = Centre_hospitalisation_commune_secteur_full.pivot_table(
    index=Col_admin_zone_name,
    columns='secteur'
    )
    Centre_hospitalisation_commune_secteur_full.columns = ['_'.join([str(i) for i in col]).strip() \
        if isinstance(col, tuple) else col for col in Centre_hospitalisation_commune_secteur_full.columns]
    # Aggéger la population
    Centre_hospitalisation_commune_secteur_full.reset_index(inplace=True)
    Centre_hopital_commune_pop=Centre_hospitalisation_commune_secteur_full.merge(pop_zone_admin, left_on=Col_admin_zone_name, \
        right_on=Col_admin_zone_name+str('_nom'), how='left')
    # On calcul maintenant les taux de couverture 
    Centre_hopital_commune_pop['lits_total_nbr_Public (pourcent / tot)'] =100* \
        Centre_hopital_commune_pop['lits_total_nbr_Public']/(Centre_hopital_commune_pop['lits_total_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_total_nbr_Prive'].sum(axis=0)) 
    Centre_hopital_commune_pop['lits_total_nbr_Prive (pourcent / tot)'] = 100*\
        Centre_hopital_commune_pop['lits_total_nbr_Prive']/(Centre_hopital_commune_pop['lits_total_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_total_nbr_Prive'].sum(axis=0))
    Centre_hopital_commune_pop['lits_bon_nbr_Public (pourcent / tot)'] =100* \
        Centre_hopital_commune_pop['lits_bon_nbr_Public']/(Centre_hopital_commune_pop['lits_bon_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_bon_nbr_Prive'].sum(axis=0)) 
    Centre_hopital_commune_pop['lits_bon_nbr_Prive (pourcent / tot)'] = 100*\
        Centre_hopital_commune_pop['lits_bon_nbr_Prive']/(Centre_hopital_commune_pop['lits_bon_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_bon_nbr_Prive'].sum(axis=0))
    Centre_hopital_commune_pop['lits_maternite_nbr_Public (pourcent / tot)'] = 100*\
        Centre_hopital_commune_pop['lits_maternite_nbr_Public']/(Centre_hopital_commune_pop['lits_maternite_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_maternite_nbr_Prive'].sum(axis=0)) 
    Centre_hopital_commune_pop['lits_maternite_nbr_Prive (pourcent / tot)'] = 100*\
        Centre_hopital_commune_pop['lits_maternite_nbr_Prive']/(Centre_hopital_commune_pop['lits_maternite_nbr_Public'].sum(axis=0)+\
            Centre_hopital_commune_pop['lits_maternite_nbr_Prive'].sum(axis=0))
    # Calcul des taux pour 1000 habitants
    Centre_hopital_commune_pop['lits_bon_nbr_Public (nbr/10000) '] = \
        Centre_hopital_commune_pop['lits_bon_nbr_Public']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_bon_nbr_Prive (nbr/10000)'] = \
        Centre_hopital_commune_pop['lits_bon_nbr_Prive']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_total_nbr_Public (nbr/10000) '] = \
        Centre_hopital_commune_pop['lits_total_nbr_Public']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_bon_nbr_Prive (nbr/10000)'] = \
        Centre_hopital_commune_pop['lits_bon_nbr_Prive']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_maternite_nbr_Public (nbr/10000)'] = \
        Centre_hopital_commune_pop['lits_maternite_nbr_Public']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_maternite_nbr_Prive (nbr/10000)'] =\
        Centre_hopital_commune_pop['lits_maternite_nbr_Prive']*10000/(Centre_hopital_commune_pop['Population'])
    Centre_hopital_commune_pop['lits_maternite_Somme (nbr/10000)']= Centre_hopital_commune_pop['lits_maternite_nbr_Public (nbr/10000)'] +\
         Centre_hopital_commune_pop['lits_maternite_nbr_Prive (nbr/10000)']
    Centre_hopital_commune_pop['lits_bon_nbr_Somme (nbr/10000)']= Centre_hopital_commune_pop['lits_bon_nbr_Public (nbr/10000) '] +\
         Centre_hopital_commune_pop['lits_bon_nbr_Prive (nbr/10000)']
    Centre_hopital_commune_pop['lits_total_nbr_Somme (nbr/10000)']= Centre_hopital_commune_pop['lits_total_nbr_Public (nbr/10000) '] +\
         Centre_hopital_commune_pop['lits_bon_nbr_Prive (nbr/10000)']
    return Centre_hopital_commune_pop

Bilan_commune=Aggregation_sectoriel_et_geo(Centre_hospitalisation,pop_commune,'commune',commune)
Bilan_prefecture=Aggregation_sectoriel_et_geo(Centre_hospitalisation,pop_prefecture,'prefecture',prefecture)
Bilan_region=Aggregation_sectoriel_et_geo(Centre_hospitalisation,pop_region,'region',region)

Bilan_commune=Bilan_commune.merge(commune[['region_nom','prefecture_nom','commune_nom','geometry']], left_on='commune', right_on='commune_nom', how='left')
Bilan_prefecture=Bilan_prefecture.merge(prefecture[['region_nom','prefecture_nom','geometry']], left_on='prefecture', right_on='prefecture_nom', how='left')
Bilan_region=Bilan_region.merge(region[['region_nom','geometry']], left_on='region', right_on='region_nom', how='left')

Bilan_commune.drop(columns='geometry').to_excel("Bilan_commune.xlsx", index=False)
Bilan_prefecture.drop(columns='geometry').to_excel("Bilan_prefecture.xlsx", index=False)
Bilan_region.drop(columns='geometry').to_excel("Bilan_region.xlsx", index=False)

############################################################################################################
#
#   Visualisation
#
############################################################################################################
def print_pop_dif_over_zone(mean_agg,field_name,Centre_soins,title,cmap_bounds="None"):
    marker_list = ['o', 's', '^', 'D', 'v', 'P', 'P', 'X', '<', '>','d','^','h']
    cmap_marker=plt.cm.get_cmap('RdGy',Centre_lits['etablissement_type'].nunique())
    Fontsize_title=20
    Fontsize_label=18
    Fontsize_tick_cbar=16
    if cmap_bounds=="None":
        vmin, vmax = mean_agg[field_name].min(), mean_agg[field_name].max()  #
    else :
        vmin, vmax = cmap_bounds[0], cmap_bounds[1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"
    fig, ax = plt.subplots(figsize=(18, 15))
    mean_agg.plot(ax=ax, cmap=cmap, norm=norm,column=field_name,alpha=0.7)
    for ind,element in enumerate(Centre_lits['etablissement_type'].unique()):
        Centre_soins[Centre_soins['etablissement_type']==element].plot(ax=ax,color=cmap_marker(ind),marker=marker_list[ind],markersize=50,edgecolor='white',label=element)
    region.plot(ax=ax,facecolor="None",edgecolor="black")
    plt.legend(loc='upper left', bbox_to_anchor=(0.60, 0.95),fontsize=Fontsize_label,frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.65, 0.98, title,  # texte
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
    cbar.set_label("lits/10000 habitants",fontsize=Fontsize_label)
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

Bilan_prefecture=gpd.GeoDataFrame(Bilan_prefecture,geometry='geometry',crs=prefecture.crs)
Bilan_region=gpd.GeoDataFrame(Bilan_region,geometry='geometry',crs=region.crs)
print_pop_dif_over_zone(Bilan_prefecture,'lits_bon_nbr_Somme (nbr/10000)',Centre_hospitalisation[Centre_hospitalisation['lits_bon_nbr']!=0],title="Nombre de lits pour 10000 habitants\n(Public & Privé)")
print_pop_dif_over_zone(Bilan_region,'lits_bon_nbr_Somme (nbr/10000)',Centre_hospitalisation[Centre_hospitalisation['lits_bon_nbr']!=0],title="Nombre de lits pour 10000 habitants\n(Public & Privé)")
print_pop_dif_over_zone(Bilan_prefecture,'lits_maternite_Somme (nbr/10000)',Centre_hospitalisation[Centre_hospitalisation['lits_maternite_nbr']!=0],title="Nombre de lits de maternité pour\n10000 habitants (Public & Privé)")
print_pop_dif_over_zone(Bilan_region,'lits_maternite_Somme (nbr/10000)',Centre_hospitalisation[Centre_hospitalisation['lits_maternite_nbr']!=0],title="Nombre de lits de maternité pour\n10000 habitants (Public & Privé)")
Bilan_prefecture[(Bilan_prefecture['lits_total_nbr_Public']==0) & (Bilan_prefecture['lits_total_nbr_Prive']==0)]

## Total
Bilan_region['lits_total_nbr_Public'].sum(axis=0)*10000/8095000

#########################################################################################################
#
#   Merging de centre sur les zones urbaines (script Basile)
#
#########################################################################################################
########### 2. Chargement et traitement des données ############
from sklearn.cluster import DBSCAN
df_urbain = gpd.read_file("./map/hexagones_classifies.gpkg")
# "final_class" ou "final_class_inseed" ou "classe" voir gdoc
df_urbain = df_urbain[df_urbain["final_class"] == "urbain"]
df_urbain["centroid"] = df_urbain.geometry.centroid
coords = np.array(list(df_urbain.centroid.apply(lambda point: (point.x, point.y))))
# Clustering spatial (distance en mètres)
MAX_DISTANCE = 2_000
db = DBSCAN(eps=MAX_DISTANCE, min_samples=1).fit(
    coords
)  # les polygones sont de 500m de large
df_urbain["cluster"] = db.labels_

# on va regrouper les polygones par cluster, le pavage de l'espace
# n'étant pas parfait, il peut y avoir des trous entre les polygones
# on dilate donc les polygones avant de les fusionner
df_urbain["geometry"] = df_urbain.geometry.apply(lambda x: x.buffer(1000))

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

marker_list = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>','d','^','h']
cmap_marker=plt.cm.get_cmap('hsv',Centre_lits['etablissement_type'].nunique())
fig, ax = plt.subplots(figsize=(15, 15))
region.plot(ax=ax,facecolor="None",edgecolor='black')
for ind,element in enumerate(Centre_lits['etablissement_type'].unique()):
    Centre_lits[Centre_lits['etablissement_type']==element].plot(ax=ax,color=cmap_marker(ind),marker=marker_list[ind],markersize=25,label=element)

df_final.plot(ax=ax,cmap='viridis',alpha=0.5,legend=True)
ax = plt.gca()
ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
ctx.add_basemap(ax, crs=Centre_lits.crs,attribution=False)
ax.set_title("Typologie des hopitaux avec des lits d'hospitalisation")
plt.show()

##################################################################################################
# maintenant je veux que les hopitaux qui sont dans une aire urbaine soit regroupés et que leur capacité 
# daccueil en terme de lits_bon_nbr soit additionnée

# je vais donc faire un merge entre les deux geodataframe
df_final = df_final.to_crs(epsg=32631)
Centre_hospitalisation_nonnul = Centre_hospitalisation_nonnul.to_crs(epsg=32631)

0.0
Centre_hospitalisation_nonnul['geometry'] = Centre_hospitalisation_nonnul.geometry.centroid
df_final.reset_index(inplace=True)
Centre_regroup_zone_urbaine = gpd.sjoin(
    Centre_hospitalisation_nonnul,
    df_final[['cluster', 'geometry']],
    how="left",
    predicate="intersects",
)
# On doit fournir des index aux cluster qui n'ont pas été regroupés avant l'agrégation
Centre_regroup_zone_urbaine = Centre_regroup_zone_urbaine.copy()
# 1. Récupérer les valeurs déjà existantes
clusters_existants = Centre_regroup_zone_urbaine['cluster'].dropna().unique().astype(int)
# 2. Nombre de NaN à combler
nouvelles_lignes = Centre_regroup_zone_urbaine['cluster'].isna().sum()
# 3. Générer des identifiants uniques non utilisés
max_cluster = clusters_existants.max() if len(clusters_existants) > 0 else 0
new_clusters = np.arange(max_cluster + 1, max_cluster + 1 + nouvelles_lignes)
# 4. Remplacer les NaN par les nouveaux identifiants
Centre_regroup_zone_urbaine.loc[Centre_regroup_zone_urbaine['cluster'].isna(), 'cluster'] = new_clusters
Centre_regroup_zone_urbaine['cluster'] = Centre_regroup_zone_urbaine['cluster'].astype(int)

## On calcul le barycentre des points
Centre_regroup_zone_urbaine['x'] = Centre_regroup_zone_urbaine.geometry.x
Centre_regroup_zone_urbaine['y'] = Centre_regroup_zone_urbaine.geometry.y
# 2. Grouper par cluster et calculer le barycentre (moyenne)
centroides = Centre_regroup_zone_urbaine.groupby('cluster')[['x', 'y']].mean().reset_index()
centroides['points'] = centroides.apply(lambda row: Point(row['x'], row['y']), axis=1)
Centre_regroupe_zone_urbaine= Centre_regroup_zone_urbaine.merge(centroides[['cluster', 'points']], on='cluster', how='inner')


Centre_regroupe_zone_urbaine = Centre_regroupe_zone_urbaine.groupby('cluster').agg(
    lits_bon_nbr=('lits_bon_nbr', 'sum'),
    lits_maternite_nbr=('lits_maternite_nbr', 'sum'),
    lits_total_nbr=('lits_total_nbr', 'sum'),
    points=('points', 'first'),
    nom_groupe=('nom_fs',list)
)
Centre_regroupe_zone_urbaine=gpd.GeoDataFrame(Centre_regroupe_zone_urbaine, geometry='points', crs=Centre_hospitalisation_nonnul.crs)
Centre_regroupe_zone_urbaine=Centre_regroupe_zone_urbaine.sjoin(commune[['region_nom','prefecture_nom','commune_nom','geometry']],predicate="within", how='inner')
Centre_regroupe_zone_urbaine.drop(columns='index_right',inplace=True)

marker_list = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>','d','^','h']
cmap_marker=plt.cm.get_cmap('hsv',Centre_lits['etablissement_type'].nunique())
fig, ax = plt.subplots(figsize=(15, 15))
region.plot(ax=ax,facecolor="None",edgecolor='black')
for ind,element in enumerate(Centre_lits['etablissement_type'].unique()):
    Centre_lits[Centre_lits['etablissement_type']==element].plot(ax=ax,color=cmap_marker(ind),marker=marker_list[ind],markersize=25,label=element)

Centre_regroupe_zone_urbaine.plot(ax=ax,markersize=100,alpha=0.3,color="yellow",legend=True)
ax = plt.gca()
ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
ctx.add_basemap(ax, crs=Centre_lits.crs,attribution=False)
ax.set_title("Typologie des hopitaux avec des lits d'hospitalisation")
plt.show()

######################################################################################################
#   Test pour améliorer le clipping
#
from geovoronoi import voronoi_regions_from_coords, points_to_coords
import matplotlib.pyplot as plt

geom_Togo=gpd.GeoDataFrame({'geometry' : [region.geometry.union_all()]}, geometry='geometry', crs=region.crs)
# 1. Obtenir les coordonnées des points
coords = points_to_coords(Centre_regroupe_zone_urbaine.geometry)
# 2. Créer une seule géométrie de découpe (le Togo)
region_shape = geom_Togo.unary_union  # shapely Polygon/MultiPolygon
# 3. Générer les régions Voronoï dans la limite
poly_shapes, pts_assignments = voronoi_regions_from_coords(coords, region_shape)
# 4. Convertir les polygones en GeoDataFrame
gdf_voronoi = gpd.GeoDataFrame({'geometry': list(poly_shapes.values())},geometry='geometry', crs=geom_Togo.crs)
# 5. Visualisation directe
Fontsize_title=20
Fontsize_label=18
Fontsize_tick_cbar=16
fig, ax = plt.subplots(figsize=(15, 15))
gdf_voronoi.plot(ax=ax, facecolor="None", edgecolor="black")
Centre_regroupe_zone_urbaine.plot(ax=ax,marker='*',markersize=50,alpha=0.8,color="red",legend=True,
                                  label="Centres hospitaliers et\nregroupements d'hôpitaux")

plt.legend(loc='lower left', bbox_to_anchor=(-0.5, 0.),fontsize=Fontsize_label,frameon=False)
plt.title("Découpage du territoire en fonction\n des hôpitaux existant",fontsize=Fontsize_title)
plt.axis('off')
plt.show()

######################################################################################
#
#   Aggrégation sur le diagramme de voronoï
#
######################################################################################

pop_hex['centroid']=pop_hex.geometry.centroid
pop_hex=pop_hex.set_geometry('centroid')
pop_hex['pop_percent']=pop_hex['total_pop_meta']/(pop_hex['total_pop_meta'].sum(axis=0))
pop_hex['population']=8095502*pop_hex['pop_percent']
Hopital_voronoi=Centre_regroupe_zone_urbaine.sjoin(gdf_voronoi, how='right', predicate='within')
Hopital_voronoi=Hopital_voronoi.sjoin(pop_hex,predicate='intersects',how='inner')
Hopital_lits_voronoi=Hopital_voronoi.groupby(['cluster']).agg(
    Population =('population','sum'),
    lits_total_nbr=('lits_total_nbr','first'),
    lits_bon_nbr=('lits_bon_nbr','first'),
    nom_groupe=('nom_groupe','first'),
    lits_maternite_nbr=('lits_maternite_nbr','first'),
    geometry=('geometry','first'),
    prefecture_nom=('prefecture_nom_left','first')
)

Hopital_lits_voronoi['groupe']=None
for idx, row in Hopital_lits_voronoi.iterrows():
    Hopital_lits_voronoi.loc[idx,'groupe']=len(row['nom_groupe'])

##### Chargement des noms de ville
import osmnx as ox
polygon = Polygon([
    (-0.26, 5.81),  # Point 1 (longitude, latitude)
    (2.0, 5.81),  # Point 2
    (2.0, 11.67),  # Point 3
    (-0.26, 11.67),  # Point 4
    (-0.26, 5.81)   # Retour au point de départ
])
tags = {"place": ["city"]}
villes = ox.features_from_polygon(polygon, tags)


Hopital_lits_voronoi['lits_par_10000hab']=Hopital_lits_voronoi['lits_total_nbr']*10000/Hopital_lits_voronoi['Population']
Hopital_lits_voronoi=gpd.GeoDataFrame(Hopital_lits_voronoi,geometry=Hopital_lits_voronoi.geometry,crs=Hopital_voronoi.crs)
villes=villes.to_crs(Hopital_lits_voronoi.crs)
villes=villes.sjoin(geom_Togo,how='inner', predicate='within')
villes_unique = villes.drop_duplicates(subset="name", keep="first")


Fontsize_title=20
Fontsize_label=18
Fontsize_tick_cbar=16

vmin, vmax = Hopital_lits_voronoi['lits_par_10000hab'].min(), Hopital_lits_voronoi['lits_par_10000hab'].max()  #
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"

fig, ax = plt.subplots(figsize=(15, 15))
Hopital_lits_voronoi.plot(ax=ax, cmap=cmap, norm=norm,column="lits_par_10000hab",alpha=0.7)
for idx, row in villes_unique.iterrows():
    geom = row.geometry
    name = row.get("name", None)
    if name and geom:
        # Choisir un point de référence : centroïde si c'est un polygone, sinon le point lui-même
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y+5000
        else:
            x, y = geom.centroid.x, geom.centroid.y+5000
        # Afficher le nom légèrement au-dessus du point
        ax.text(x, y + 1000, name, fontsize=12, ha='center', va='bottom', color='black')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
Centre_regroupe_zone_urbaine.plot(ax=ax,marker='*',markersize=50,alpha=0.8,color="black",legend=True,
                                  label="Centres hospitaliers et\nregroupements d'hôpitaux")

plt.legend(loc='lower left', bbox_to_anchor=(-0.5, 0.),fontsize=Fontsize_label,frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(0.55, 0.98, "Lits d'hospitation pour 10000 habitants en tenant\ncompte de la distance aux hopitaux",
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=Fontsize_title)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
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
cbar.set_label("lits/10000 habitants",fontsize=Fontsize_label)
cbar.ax.tick_params(labelsize=Fontsize_tick_cbar)
rect = Rectangle(
    (0.21, 0.115), 0.6, 0.81,                        # position et taille (x, y, width, height)
    linewidth=2,                        # épaisseur du trait
    edgecolor='black',                  # couleur de bord
    facecolor='none',                   # transparent à l'intérieur
    transform=fig.transFigure,          # important : coordonnées en fig (et non ax)
    zorder=1000                         # bien au-dessus de tout le reste
)

fig.patches.append(rect)
plt.show()


Hopital_lits_voronoi.drop(columns='geometry').to_excel("Bilan_zone_chalandise.xlsx", index=False)



####


Tableau_final = pd.read_excel('./Bilan_déploiement_prefectoral_final.xlsx')
Tableau_final=Tableau_final.merge(prefecture,left_on ='Préfectures',right_on='prefecture_nom',how='inner')
Tableau_final=gpd.GeoDataFrame(Tableau_final,geometry='geometry',crs=prefecture.crs)
Fontsize_title=20
Fontsize_label=18
Fontsize_tick_cbar=16
vmin, vmax = Tableau_final["dep1"].min(), 200  #
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(figsize=(15, 15))
Tableau_final.plot(ax=ax, cmap=cmap, norm=norm,column="dep1",alpha=0.7)
region.plot(ax=ax,facecolor="None",edgecolor="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(0.67, 0.98, "Déploiement de lits d'hôpitaux\nau niveau préfectoral",  # texte
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
cbar.set_label("Nombre de lits à déployer",fontsize=Fontsize_label)
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




marker_list = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>','d','^','h']
cmap_marker=plt.cm.get_cmap('hsv',Centre_lits['etablissement_type'].nunique())
Fontsize_title=20
Fontsize_label=18
Fontsize_tick_cbar=16
vmin, vmax = Tableau_final["Nombre de lits pour la moyenne prefectoral"].min(), 200  #
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(figsize=(15, 15))
Tableau_final.plot(ax=ax, cmap=cmap, norm=norm,column="Nombre de lits pour la moyenne prefectoral",alpha=0.7)
# for ind,element in enumerate(Centre_lits['etablissement_type'].unique()):
#     Centre_lits[Centre_lits['etablissement_type']==element].plot(ax=ax,color=cmap_marker(ind),marker=marker_list[ind],markersize=25,label=element)

region.plot(ax=ax,facecolor="None",edgecolor="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)



ax.text(0.67, 0.98, "Déploiement de lits d'hôpitaux\nau niveau préfectoral",  # texte
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
cbar.set_label("Nombre de lits à déployer",fontsize=Fontsize_label)
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



Tableau_final = pd.read_excel('./Bilan_déploiement_prefectoral_final.xlsx')
Tableau_final=Tableau_final.merge(prefecture,left_on ='Préfectures',right_on='prefecture_nom',how='inner')
Tableau_final=gpd.GeoDataFrame(Tableau_final,geometry='geometry',crs=prefecture.crs)
Fontsize_title=20
Fontsize_label=18
Fontsize_tick_cbar=16
vmin, vmax = Tableau_final["dep1"].min(), 200  #
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"
fig, ax = plt.subplots(figsize=(15, 15))
Tableau_final.plot(ax=ax, cmap=cmap, norm=norm,column="dep1",alpha=0.7)
region.plot(ax=ax,facecolor="None",edgecolor="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(0.67, 0.98, "Déploiement de lits d'hôpitaux\n(7.2 lits/10000hab minimum par pref.)",  # texte
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
cbar.set_label("Nombre de lits à déployer",fontsize=Fontsize_label)
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
