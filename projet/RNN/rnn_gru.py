
# coding: utf-8



#############################################################################

## Ce fichier permet d'apprendre un réseau de neurones récurrents GRU 
## avec un jeu de test équilibré et un jeu de test déséquilibré

#############################################################################


import torch
import torch.nn as nn
import os.path
import os
import random
import torch.optim as optim
from BD_Deezer import *
from fonctions_aides_rnn import *

# # Mise en forme des données

c_app=os.path.abspath(".")+"/BD/deezer/apprentissage"
c_mod=os.path.abspath(".")+"/BD/deezer/modele"
c_pred=os.path.abspath(".")+"/BD/deezer/prediction"
c_test=os.path.abspath(".")+"/BD/deezer/test"
c_svm=os.getcwd() + "/SVMlight"

# Creation d'un dictionnaire qui à chaque genre associe une liste de liste contenant un temps, une ville et un nombre d'ecoute <br/>
# Creation d'un ensemble contenant toutes les villes du fichier et d'un ensemble contenant toutes les dates <br/>
# Creation d'un dictionnaire associant les identifiants des albums à leur genre, d'un dictionnaire associant les genre aux albums et enfin un ensemble contenant les identifiants des albums
d_a,set_ville, set_date = lecture_json_album(BD_a)
d_g, d_g_i, set_id_a = lecture_json_genre(BD_g)

a = d_a.copy()
g = d_g.copy()
a = all_conc_album(a)

dates = list(set_date)
villes = list(set_ville)
albums = list(set_id_a)


# Creation d'un dictionnaire qui à chaque groupement de genre associe les identifiants des albums correspondant.
d_g_i_g = groupe(d_g_i, group)
nb_class=len(d_g_i_g.keys())

# écriture d'un jeu d'apprentissage équilibré, d'un jeu d'apprentissage déséquilibré 
# et d'un jeu de test déséquilibré dans des fichiers, cest jeu seront réutilisé dans les rnn
creation_train(albums, nb_class, d_g_i_g)


#regroupement de dates par blocs de 10
dates_bloc=date_to_block(dates)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)            
        
    def forward(self, input, hidden):
        
        self.input_len = len(input)                   
        combined = torch.cat((input, hidden), 1)     
        
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self, input):
        return torch.zeros(len(input), self.hidden_size)  



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)             
        
    def forward(self, input, hidden):
        
        self.input_len = len(input)                           
        output, hidden = self.gru(input, hidden)
        ## from (1, N, hidden) to (N, hidden)
        rearranged = hidden.view(hidden.size()[1], hidden.size(2))
        output = self.linear(rearranged)
        output = self.softmax(output)
        return output
    
    def initHidden(self, input):
        return torch.zeros(1, len(input[0]), self.hidden_size)  



def train_rnn(a, train, test, villes, d_g_i_g, epsilon, dates_bloc, nb_class, titre_f):
    """ a : dictionnaire de la forme {id_a : {tps : {ville : nb_ecoute}}}
        app : liste d'identifiants d'album constituant l'ensemble d'apprentissage
        test : liste d'identifiants d'album constituant l'ensemble de test
        villes : listes des villes 
        d_g_i_g : dictionnaire qui à chaque groupement de genre associe les identifiants des albums correspondant
        epsilon : seuil d'erreur en dessous duquel le rnn arrete d'apprendre
        dates_blocs : liste de 36 listes contenant 10 temps consécutif chacune
        nb_class : nombre de genres différents
        titre_f : titre du fichier dans lequel les résultats seront écrits
        Fonction qui réalise l'entrainement d'un rnn GRU 
    """
    
    #création du réseau
    fichier = open(titre_f , "w")
    fichier.close()

    l=1000
    itt=0
    
    score_test=[]
    score_train=[]
    erreur=[]
    
    #initilisation du rnn
    rnn = RNN(len(villes), len(villes), nb_class)
    optimizer = optim.Adam(rnn.parameters(), lr=0.00008)      
    criterion = nn.CrossEntropyLoss()
    
    #tant que l'erreur n'est pas inféreur à un epsilon passé en paramètre
    while(l>epsilon):
        #on remet les paramètres à zéro
        optimizer.zero_grad()
        
        tr_minibatch = random.sample(list(app), 50)
        
        #pour chaque pas de temps
        for i_d in range(len(dates_bloc)): 
          #on crée notre base d'apprentissage à ce temps là 
            input, label = transfo_base_rnn(a, dates_bloc[i_d], tr_minibatch, villes, d_g_i_g)
            
            if (i_d==0):
                #si c'est le premier pas de temps on initialise hidden à zéro 
                hidden = rnn.initHidden(input)
            #on fait tourner notre réseau
            output, hidden = rnn(input, hidden)

        #une fois qu'on a parcouru tous les temps on calcul l'erreur
        loss = criterion(output, label)
        #on fait un backward sur l'erreur 
        loss.backward()
        
        #on modifit les paramètre du réseau
        optimizer.step()
        l=loss.item()
        
        #ecriture ponctuelle dans fichier sauvegarde
        if(itt%30==0):
            fichier=  open(titre_f, "a")
            fichier.write("iteration: "+ str(itt)+"\n")
            fichier.write("loss: "+str (l)+"\n")
            erreur.append(l)
            #on ajoute dans le fichier le score du train             
            _,predicted=torch.max(output.data,1)        
            total = label.size(0)
            correct = (predicted == label).sum().item()
            ap=correct/total
            fichier.write("score train: "+ str(ap)+"\n")
            score_train.append(ap)
            #on ajoute dans le fichier le score du test
            with torch.no_grad():
                for i_d in range(len(dates_bloc)):  
                    input, label = transfo_base_rnn(a, dates_bloc[i_d], test, villes, d_g_i_g)
                    if (i_d==0):
                        #si c'est le premier pas de temps on initialise hidden à zéro 
                        hidden = rnn.initHidden(input)
                        #on fait tourner notre réseau
                    output, hidden = rnn(input, hidden)
                _, predicted = torch.max(output.data, 1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
            te=correct/total
            fichier.write("score test: "+ str(te)+"\n\n")
            score_test.append(te)
            fichier.close()
            
        itt+=1
              
    return score_train, score_test, erreur


def train_gru(a, app, test, villes, d_g_i_g, epsilon, dates_bloc, nb_class, titre_f):
    """ a : dictionnaire de la forme {id_a : {tps : {ville : nb_ecoute}}}
        app : liste d'identifiants d'album constituant l'ensemble d'apprentissage
        test : liste d'identifiants d'album constituant l'ensemble de test
        villes : listes des villes 
        d_g_i_g : dictionnaire qui à chaque groupement de genre associe les identifiants des albums correspondant
        epsilon : seuil d'erreur en dessous duquel le rnn arrete d'apprendre
        dates_blocs : liste de 36 listes contenant 10 temps consécutif chacune
        nb_class : nombre de genres différents
        titre_f : titre du fichier dans lequel les résultats seront écrits
        Fonction qui réalise l'entrainement d'un rnn GRU 
        """
    #création du réseau
    fichier = open(titre_f, "w")
    fichier.close()
    score_test=[]
    score_train=[]
    erreur=[]

    l=1000
    itt=0
    
    #initilisation du gru
    gru = GRU(input_size=len(villes), hidden_size=len(villes), output_size=nb_class)
    optimizer = optim.Adam(gru.parameters(), lr=0.0001)      
    criterion = nn.CrossEntropyLoss()
    
    #tant que l'erreur n'est pas inféreur à un epsilon passé en paramètre
    while(l>epsilon):
        #on remet les paramètres à zéro
        optimizer.zero_grad()
        
        tr_minibatch = random.sample(list(app), 200)
        
       
        #on crée notre base d'apprentissage à ce temps là 
        input, label = transfo_base_gru(a, dates_bloc, tr_minibatch, villes, d_g_i_g) 

        hidden = gru.initHidden(input)             
        #on fait tourner notre réseau
        output= gru(input, hidden)

        #une fois qu'on a parcouru tous les temps on calcul l'erreur
        loss = criterion(output, label)
        #on fait un backward sur l'erreur 
        loss.backward()
        
        #on modifit les paramètre du réseau
        optimizer.step()
        l=loss.item()
        
        #ecriture ponctuelle dans fichier sauvegarde
        if(itt%50==0):
            fichier=  open(titre_f, "a")
            fichier.write("iteration: "+ str(itt)+"\n")
            fichier.write("loss: "+str (l)+"\n")
            erreur.append(l)
            #on ajoute dans le fichier le score du train 
            _,predicted=torch.max(output.data,1)        
            total = label.size(0)
            correct = (predicted == label).sum().item()
            ap=correct/total
            fichier.write("score train: "+ str(ap)+"\n")
            score_train.append(ap)
            #on ajoute dans le fichier le score du test
            input, label = transfo_base_gru(a, dates_bloc, test, villes, d_g_i_g) 
            hidden = gru.initHidden(input)
            output= gru(input, hidden)
            _,predicted=torch.max(output.data,1)        
            correct = (predicted == label).sum().item()
            te=correct/total
            fichier.write("score test: "+ str(te)+"\n\n")
            score_test.append(te)
            fichier.close()
                      
        itt+=1
          
    return score_train, score_test, erreur

#entrainement d'un rnn basic sur un jeu de données déséquilibré
test= lecture_ensemble("ensemble_test.txt")
app=lecture_ensemble("ensemble_a.txt")
#score_train, score_test, erreur = train_rnn(a, app, test, villes, d_g_i_g, 0.2, dates_bloc, nb_class,"res_rnn_desequilibre.txt")

#entrainement d'un rnn basic sur un jeu de données équilibré
app_eq=lecture_ensemble("ensemble_a_egalitaire.txt")
#score_train, score_test, erreur = train_rnn(a, app_eq, test, villes, d_g_i_g, 0.2, dates_bloc, nb_class,"res_rnn_equilibre.txt")

#entrainement d'un rnn gru sur un jeu de données déséquilibré
#score_train, score_test, erreur= train_gru(a, app, test, villes, d_g_i_g, 0.1, dates_bloc, nb_class, "res_gru_desequilibre.txt")

#entrainement d'un rnn gru sur un jeu de données équilibré
score_train, score_test, erreur= train_gru(a, app_eq, test, villes, d_g_i_g, 0.1, dates_bloc, nb_class, "res_gru_equilibre.txt")


