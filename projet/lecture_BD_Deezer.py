
#!/usr/bin/env python
# coding: utf-8

# # BD Deezer

# In[2]:

#from apprentissage_test import *
from pprint import pprint
import numpy as np
import random
import json
import math


# In[3]:

BD_g = "./BD/deezer/album_genres_new_releases.json"
BD_a = "./BD/deezer/sampled_geoloc_counters_new_releases.json"
nb_block = 10


# ### Lecture des fichiers de données Deezer

# In[4]:

def lecture_json_genre(fichier):
    """fichier : nom du fichier à lire au format json
       renvoie un ditionnaire (id:genre) et un dictionnaire (genre:[liste identifiant]), ainsi qu'un ensemble 
       d'identifiants d'albums existant"""
    
    d = dict()
    d_g_i = dict()
    set_id_a = set()
    for line in open(fichier, 'r'):
        l = json.loads(line)
        set_id_a.add(l['album_id'])
        if(l['album_id'] not in d.keys()):
            d[l['album_id']] = []
        d[l['album_id']].append(l['genre_name'])
        
        if(l['genre_name'] not in d_g_i.keys()):
            d_g_i[l['genre_name']] = []
        d_g_i[l['genre_name']].append(l['album_id'])
            
    return d, d_g_i, set_id_a


# In[5]:

def lecture_json_album(fichier):
    """fichier : nom du fichier à lire au format json
       renvoie un dictionnaire (id_album : [[date, ville, nb_vue]]), un ensemble de ville et de date qui existe dans 
       ce fichier"""
    
    d = dict()
    set_ville = set()
    set_date = set()
        
    for line in open(fichier, 'r'):
        l = json.loads(line)
        del l['age_group']
        del l['platform_name']
        del l['nstreams']
        ai = l['album_id']
        del l['album_id']
        set_ville.add(l['loc_city'])
        set_date.add(l['d'])
        
        if(ai not in d):
            d[ai] = []
        d[ai].append(l)
        
    return d, set_ville, set_date

group = [['Rap français'],
         ['Rap/Hip Hop', 'East Coast'], 
         ['Pop', 'Pop internationale', 'Afro Pop', 'Pop latine', 'Pop Indé', 'Pop indé/Folk'],
         ['Dance', 'Disco', 'Dancehall/Ragga', 'Singer & Songwriter'],
         ['Jazz', 'Blues', 'Country', 'Soul', 'Reggae', 'Soul & Funk'],
         ['R&B'],
         ['Electro', 'Techno/House', 'Electro Pop/Electro Rock'],
         ['Música colombiana', 'Chanson française', 'World', 'Latino', 'Musique africaine', 'Variété Internationale', 'Musique arabe', 'Musique brésilienne'],
         ['Alternative', 'Chill Out/Trip-Hop/Lounge', 'Trance', 'Dub'],
         ['Films/Jeux vidéo', 'Jeunesse', 'Sports'],
         ['Rock', 'Rock indé', 'Hard Rock', 'Rock & Roll/Rockabilly', 'Folk', 'Rock français', 'Metal'],
         ['Classique', 'Opéra', 'Musique religieuse',  'Gospel', 'Musiques de films', 'Comédies musicales']]


# In[8]:

def groupe(d_g_i, group):
    """d_g_i : dictionnaire (genre : [liste identifiant])
       group : liste du regroupement de genre
       renvoie un dicionnaire qui pour chaque groupement de genre de 'groupe' associe une liste d'identifiant d'albums"""
    d = dict()
    a=1
    d[0] = []
    for gi in range(0,len(group)):
        tmp = []
        for g in group[gi]:
            tmp += d_g_i[g]
        if(len(tmp) > 49):
            d[a] = tmp
            a+=1
        #si nombre d'identifiants est insuffisant (<100), on ajoute la liste la liste à la classe 'autre' (0)
        else:
            d[0] += tmp
    return d


def conc_album_temps(album):
    """album est une liste de dictionnaires contenant une date, une ville, un nombre d'écoute et un nombre d'utilisateur
    Cette fonction crée un dictionnaire qui a pour clé les différentes dates d'écoute et pour valeur les dictionnaires 
    (contenant une ville et un nombre d'utilisateur) correspondant à cette date"""
    d_t = dict()
    res = dict()

    for d in album:
        date = d['d']
        del d['d']
        if(date not in d_t):
            d_t[date] = []
        d_t[date].append(d)
    
    for t,v in d_t.items():
        res[t]=conc_album_ville(v)
        
    return res


# In[11]:

def conc_album_ville(a):
    """a est une liste de dictionnaires contenant une ville, un nombre d'écoute et un nombre d'utilisateur
    Cette fonction crée un dictionnaire qui a pour clé les différentes villes et pour valeur le nombre d'utilisateur correspondant à cette ville"""
    
    d_v = dict()
    for v in a:
        ville = v['loc_city']
        if(ville not in d_v):
            d_v[ville] = 0
        d_v[ville] += v['nusers']
        
    return d_v


# In[12]:

def all_conc_album(albums):
    """albums : dictionnaire (id_album : [[date, ville, nb_vue]])
       renvoie un dictionnaire (id_album : [date : [ville : nb_vue]])"""
    dico_a = dict()
    for i, a in albums.items():
        dico_a[i] = conc_album_temps(a)
    return dico_a


# In[13]:

def cross_validation_groupe(i_liste_id, nb_block, liste_id_a):
    """i_liste_id : liste d'identifiant d'un genre (groupe)
       nb_block : nombre de block que l'on veut au mieux, sinon on réduit automatiquement
       list_id_a : ensemble des identifiants d'albums existant
       renvoie une liste train de la forme [genre, train (train1, train2...)] et une liste test de 
       la forme [genre, test (test1, test2...)]"""
    block_test = []
    block_train = []
    if(len(i_liste_id)/nb_block == 0):
        tmp_liste = i_liste_id.copy()
        for i in range(len(i_liste_id)):
            block_test.append([i_liste_id[i]] + random.sample(liste_id_a, 1))
            del tmp_liste[i]
            
            block_train[i].append(tmp_liste + random.sample(liste_id_a, len(tmp_liste)))
            
    else:
        #######Reste de la division euclidienne entre len(i_liste_id) et nb_block n'est pas pris en compte dans la création des fichiers
        nb_val = len(i_liste_id)//math.floor(nb_block)
        for i in range(0, len(i_liste_id)-nb_val, nb_val):
            block_test.append(i_liste_id[i : i+nb_val] + random.sample(liste_id_a, nb_val))
            tmp_liste = i_liste_id.copy()
            del tmp_liste[i : int(i+nb_val)]
            block_train.append(tmp_liste + random.sample(liste_id_a, len(tmp_liste)))
    
    return block_test, block_train


# In[14]:

def all_cross(d_g_i_g, nb_block, liste_id_a):
    """d_g_i_g : dictionnaire des identifiants d'albums groupe par genre
       nb_block : nombre de blocks que l'on veut dans le meilleur des cas
       liste_id_a : liste de tous les identifiants d'albums existant
       renvoie pour tous les genres tous les tests et tous les trains"""
    res_train = []
    res_test = []
    for g in d_g_i_g.keys():
        te, tr = cross_validation_groupe(d_g_i_g[g], nb_block, liste_id_a)
        res_train.append(tr)
        res_test.append(tr)
    return res_train, res_test


def format_ligne(a, id_album, g, l_ville, l_date, g_etude):
    """a : dictionnaire d'albums (ensemble d'exemple)
       id_album : identifiant d'un album
       g : genre de l'album id_album
       l_ville : liste des villes existante
       l_date : liste des dates existante
       g_etude : numéro du genre à étudier
       renvoie une ligne string écrite au format svmlight correspondant à un album et ces différentes dimensions"""
    if(g==g_etude):
        tmp = "1 "     #association d'un numéro pour un genre par indexation
    else:
        tmp = "-1 "
            
    dim = 0                                        #dimension dans le fichier pour SVM light
    #parcourt date
    for date in l_date:
        if(date in a[id_album]):
            
            #parcourt ville
            for ville in l_ville:
                if(ville in a[id_album][date]):
                    tmp += str(dim) + ":" + str(a[id_album][date][ville]) + " "
                dim += 1
        else:
            dim += len(l_ville)
    return tmp


# In[18]:

def ecriture_groupe(a, test, train, d_g_i_g, liste_ville, liste_date):
    """a : dictionnaire d'album
       test : liste d'itentifiant pour chaque fichier (définit par corss-validation)
       train liste d'identifiant pour chaque fichier (définit par corss-validation)
       d_g_i_g : dictionnaire des identifiants d'albums groupe par genre
       liste_ville : liste de toutes les villes existantes
       liste_date : liste de toutes les dates existantes
       écrire l'ensemble des fichiers de test et d'apprentissage lisible par svm
       le fichier apprentissage0.2.txt -> apprentissage classe 0 2e ensemble de test"""
    
    for g_etude in list(d_g_i_g.keys()):
        p = 0
        for liste_i in test[g_etude]:
            fichier_test = open("./BD/deezer/test/test" + str(g_etude) + "." + str(p) + ".txt", "w")
            p += 1
            for i in liste_i:
                t_g = recuperation_genre(d_g_i_g, i)
                for g in t_g:
                    tmp = format_ligne(a, i, g, liste_ville, liste_date, g_etude)
                    fichier_test.write(tmp+"\n")
        
        p = 0
        for liste_i in train[g_etude]:
            fichier_train = open("./BD/deezer/apprentissage/apprentissage" + str(g_etude) + "." + str(p) + ".txt", "w")
            p += 1
            for i in liste_i:
                t_g = recuperation_genre(d_g_i_g, i)
                for g in t_g:
                    tmp = format_ligne(a, i, g, liste_ville, liste_date, g_etude)
                    fichier_train.write(tmp+"\n")


# In[19]:

def recuperation_genre(d_g_i_g, id_album):
    """d_g_i_g : dictionnaire des identifiants d'albums groupé par genre
       id_album : identifiant de l'album dont on veut récupérer le genre
       renvoie la liste des genres associé à l'identifiant id_album"""
    t_g = []
    for g in list(d_g_i_g.keys()):
        if(id_album in d_g_i_g[g]):
            t_g.append(g)
            
    if(len(t_g)==0):
        print("error : l'album n'a pas de genre groupe")
    return t_g



