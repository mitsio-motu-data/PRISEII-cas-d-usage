################################################################################
# Script Name  : main.py
# Description  : Script général
# Auteur       : basile@mitsiomotu.com
# Date : 2025/04/25
################################################################################

import geopandas as gpd

from preprocessing_routes import preprocess_osm_road
from preprocessing_lampadaires import preprocess_lampadaires
from aires_urbaines import main_aires_urbaines
from r


FINAL_PATH = "./data/main.gpkg"