# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:50:17 2018

@author: alexi
"""

###################################################################################################

# Fichier qui contient plusieurs fonctions nécéssaire aux différents rnn utilisé
# dans d'autres fichiers

###################################################################################################


import numpy as np
import random
import torch
from BD_Deezer import *



def creation_train(albums, nb_class, d_g_i_g ):
    """ album: liste des identifiants des albums de la base
        nb_class: nombre de genres différents
        d_g_i_g: dictionnaire qui à chaque groupement de genre associe les identifiants des albums correspondant
        Fonction qui écrit des iddentifiant d'album dans 3 fichiers de tel sorte qu'on ai un fichier contenant
        des identifiants d'albums constituant jeu d'apprentissage déséquilibré, un autre des id d'album constituant 
        un jeu d'apprentissage équilibré et le dernier constituant un jeu de test déséquilbré
    """
    fichier_t=open("ensemble_test.txt","w")
    fichier_a_e = open("ensemble_a_egalitaire.txt","w")
    fichier_a=open("ensemble_a.txt","w")
    train_e=[]
    taille_train=0
    test=[]
    for g in d_g_i_g.keys():
        a=random.sample(d_g_i_g[g], 40)
        taille_train += len(a)
        train_e.append(a)
        
    test=random.sample(albums,100)
    train=random.sample(albums,taille_train)
    train_e=list(np.reshape(np.array(train_e), taille_train))

    fichier_t.write(str(test))
    fichier_a_e.write(str(train_e))
    fichier_a.write(str(train))
    
    

def lecture_ensemble(n_fichier):
    """ fonction qui permet de lire la liste qui se trouve dans le fichier passé en argument"""
    res=open(n_fichier).read()
    res=res[1:-1]
    res=res.split(", ")
    res=[int(i) for i in res]
    return res


def date_to_block(liste_dates):
    res=[]
    for i in range(36):
        tmp=[]
        if ( i<35): 
            for j in range(10):
                tmp.append(liste_dates[i*10+j])
            res.append(tmp)
        else:
            for j in range(8):
                tmp.append(liste_dates[i*10+j])
            res.append(tmp)
    return res


def transfo_base_rnn(a, bloc_dates, sous_base, villes, d_g_i_g):
    """Transforme une base de la forme {id_a : {tps : {ville : nb_ecoute}}} en tableau representant l'input de notre RNN au
       temps t c'est-a-dire le nombre d'ecoute de chaque album dans chaque ville au temps t"""
    
    tab = []
    label = []

    
    for i in range(len(sous_base)):
        i_a=sous_base[i]
        genres=recuperation_genre(d_g_i_g, i_a)
        for g in genres:
            label.append(g)
            tmp=np.zeros(len(villes))
            for t in bloc_dates:
                #print(a)
                if (t in a[i_a]):
                    for i_v in range(len(villes)):
                        v=villes[i_v]
                        if(v in a[i_a][t]):
                            tmp[i_v] += a[i_a][t][v]
            tab.append(tmp)

            
    return torch.tensor(tab,dtype=torch.float), torch.tensor(label, dtype=torch.long)

def transfo_base_gru(a, bloc_dates, sous_base, villes, d_g_i_g):
    """Transforme une base de la forme {id_a : {tps : {ville : nb_ecoute}}} en tableau representant l'input de notre GRU au
       temps t c'est-a-dire le nombre d'ecoute de chaque album dans chaque ville au temps t"""
    res=[]
    label = []
    for i_t in range(len(bloc_dates) ): 
        tab = []
        for i in range(len(sous_base)):
            i_a=sous_base[i]
            genres=recuperation_genre(d_g_i_g, i_a)
            for g in genres:
                if(i_t==0):
                    label.append(g)
                tmp=np.zeros(len(villes))
                for t in bloc_dates[i_t]:
                    if (t in a[i_a]):
                        for i_v in range(len(villes)):
                            v=villes[i_v]
                            if(v in a[i_a][t]):
                                tmp[i_v] += a[i_a][t][v]
                tab.append(tmp)
        res.append(tab)

    return torch.tensor(res,dtype=torch.float), torch.tensor(label, dtype=torch.long)