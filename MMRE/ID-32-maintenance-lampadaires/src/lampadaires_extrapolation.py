################################################################################
# Script Name  : lampadaires_extrapolation.py
# Description  : Extrapolation du diagnostic des lampadaires hors grappe
# Auteur       : basiledesj@hotmail.fr
# Date : 2025/04/23
################################################################################
"""
On a diagnostiqué les dysfonctionnements des lampadaires hors grappe avec l'analyse textuelle des commentaires PRISE, 
Tous les lampadaires n'ont pas de commentaires
On va donc extrapoler.

Les données sont traitées ici :
https://docs.google.com/spreadsheets/d/1IkYreb1YReF__BtJZsJ1ot5P_GNEmBCIAcUE5UEy8TM/edit?usp=sharing
"""
import numpy as np

from preprocessing_lampadaires import preprocess_lampadaires_prise
from lampadaires_grappes_dysfonctionnement import find_grappes


df_prise = preprocess_lampadaires_prise(keep_comments=False)
df_grappe, _ = find_grappes()

# On ne garde que les lampadaires dysfonctionnels hors grappe
df_prise = df_prise[df_prise["est_fonctionnel"] == 0].copy()
id_grappes = df_grappe.id.to_numpy()
df_prise["in_grappe"] = np.isin(df_prise["id"], id_grappes)


df_extra = df_prise[df_prise["in_grappe"] == False]

print(df_extra.groupby("type").id.count())