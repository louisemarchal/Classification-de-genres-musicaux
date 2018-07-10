
# coding: utf-8

# In[83]:

import os.path
import os
import matplotlib.pyplot as plt
import numpy as np


# In[67]:

nb_class=11
c_app=os.path.abspath(".")+"/BD/deezer/apprentissage"
c_mod=os.path.abspath(".")+"/BD/deezer/modele"
c_pred=os.path.abspath(".")+"/BD/deezer/prediction"
c_test=os.path.abspath(".")+"/BD/deezer/test"
c_svm=os.getcwd() + "/SVMlight"


def apprentissage(c_app, c_mod, c_svm, option_a=""):
    """c_app est le chemin vers les fichiers d'apprentissages
        c_mod est le chemin vers les fichiers de modeles crées par svmLight
        c_svm est le chemin menant aux commandes de SVMlight
       option_a contient les options d'apprentissage
       Cette fonction lance la commande svm_learn sur tous les fichiers d'apprentissage et crée pour chacun un fichier modele"""
    
    l=os.listdir(c_app)
    for i in range(nb_class):
        l_tmp=[k for k in l if "apprentissage"+str(i)+"." in k]
        for j in range(len(l_tmp)):
            tmp=os.popen(c_svm+"/svm_learn " + option_a + " " + c_app+"/"+l_tmp[j] + " " + c_mod+"/modele"+str(i)+"."+str(j)+".txt")
            tmp.read()

# In[77]:

def test(c_test, c_mod, c_svm, c_pred, nom_fichier,option_t=""):
    """c_test est le chemin vers les fichiers d'apprentissages
        c_mod est le chemin vers les fichiers de modeles crées par svmLight
        c_svm est le chemin menant aux commandes de SVMlight
        c_pre est le chemin vers les fichiers de prédictions crées pas svmLight
        nom_fichier correspond au début du nom des fichiers que l'on veut utiliser pour le test (test ou apprentissage)
       option_a contient les options de test
       Cette fonction lance la commande svm_classify sur tous les fichiers de test et crée pour chacun un fichier prédiction"""
    
    l_t=os.listdir(c_test)
    l_m=os.listdir(c_mod)
    score=[]
    best_model=dict()  #permet de conserver le meilleur modèle parmis ceux créer avec la cross validation
    
    for i in range(nb_class):
        l_t_tmp=[k for k in l_t if nom_fichier+str(i)+"." in k]
        l_m_tmp=[k for k in l_m if "modele"+str(i)+"." in k]
        s_tmp=[]
        for j in range(len(l_t_tmp)):
            test=c_test+"/"+l_t_tmp[j] 
            pred=c_pred+ "/prediction"+str(i)+"."+str(j)+".txt"
            tmp=os.popen(c_svm+"/svm_classify " + option_t + " " + test + " " + c_mod+"/"+l_m_tmp[j]+ " "+ pred)
            tmp.read()
            s_tmp.append(evaluation(pred, test))
        score.append(s_tmp)
        best_model[i]=np.argmax(s_tmp)
        
    return score, best_model


def evaluation(f_pred, f_test):
    """HYP : f_pred est le chemin absolu d'un fichier de prédiction 
             f_test est le chemin absolu d'un fichier de test correspondant à la prédiction faite par SVM
       Cette fonction calcul score d'un modèle"""
    #tableau des predictions
    p = []
    with open(f_pred, "r") as f:
        for line in f.readlines():
            if(float(line[:-1]) > 0):
                p.append(1)
            elif(float(line[:-1]) < 0):
                p.append(-1)
            else:
                p.append(0)

    #tableau des labels de test
    lab = []
    with open(f_test, "r") as f:
        for line in f.readlines():
            tmp=line[0:2] 
            if(tmp=="\n"): #la dernière ligne lue commence par \n
                lab.append(0)
            else:
                lab.append(int(tmp))     
    
    scor = 0
    for i in range(len(p)):
        if(p[i]==lab[i]):
            scor += 1
    return scor*1.0/len(p)


# In[81]:

def app_test_acc( c_app,c_test, c_mod, c_pred, option_a="", option_t=""):
    apprentissage(c_app,c_mod,c_svm, option_a)
    score, best_model=test(c_test, c_mod, c_svm, c_pred, "test",option_t)
    score_train, b = test(c_app, c_mod, c_svm, c_pred, "apprentissage",option_t)
    return score, best_model, score_train


# In[84]:

def testeur(c_app,c_test, c_mod, c_pred, c, kernel):
    """c_app: chemin vers les fichiers d'apprentissage
       c_test: chemin vers les fichiers de test
       c_mod: chemin vers les fichiers de modélisation
       c_pred: chemin vers les fichiers de pr"diction
       c: valeurs du paramètre C
       kernel: valeur du paramètre t, indiquant le noyaux à utiliser
       """
    
 
    print("lancement avec c= "+str(c) +" et t= "+str(kernel))
    tmp, best_model, score_train = app_test_acc(c_app, c_test, c_mod, c_pred,"-c " + str(c) + " -t " + str(kernel) + " -# 300")
    sc_test = np.mean(tmp, axis=1)
    sc_train= np.mean(score_train, axis=1)
    #print("accuracy test", sc_test)
    #print("accuracy train", sc_train)
    return sc_test, sc_train


def prediction_multi_classe(c_app, c_test, c_mod, c_pred):
    """c_mod:chemin vers les fichiers de modélisation
    fonction qui calcul l'accuracy d'un fichier de test avec des exemples multi classe"""
       
    #on calcul le score du classifieur sur toutes les classes confondues

    l_m=os.listdir(c_mod)
    pred_classe=[]
    #reecriture_fichier(c_test+ "/test_all.txt", c_test + "/new_test_all.txt")

    for cl in range(nb_class):
        #on cherche tous les modèles de cette classe
        l_m_tmp=[k for k in l_m if "modele"+str(cl)+"." in k]
        pred=[]
        for j in range(len(l_m_tmp)):
            #on prédit si l'exemple appartient à la classe
            model = c_mod +"/"+ l_m_tmp[j]+" "
            p=os.popen("./SVMlight/svm_classify "+ c_test+ "/test_all.txt " + model+ c_pred+ "/pred_all.txt")
            p.read()
            p=[]
            with open(c_pred + "/pred_all.txt", "r") as f_p:
                for l in f_p.readlines():
                    p.append(float(l[:-1]))
            pred.append(p)
            
        pred=np.array(pred)
        pred=pred.T
        tmp=[]
        for i in range(len(pred)):
            tmp.append(np.mean(pred[i]))
        pred_classe.append(tmp)
    pred_classe=np.array(pred_classe)
    pred_classe=pred_classe.T
    pred_classe=np.argmax(pred_classe, axis=1)
    score=0
    cpt=0
    with open(c_test + "/test_all.txt", "r") as f:
        for t in f.readlines():
            g=float(t[:2])
            if(cpt<len(pred_classe) and float(pred_classe[cpt])==g):
                score+=1
            cpt+=1

    print("accuracy",score*1.0/cpt)
    return score*1.0/cpt
        
    
        
# In[82]:
def lancement (c_app,c_test, c_mod, c_pred, l_c, kernels):
    train = open(c_app + "/app_acc.txt", "w")
    test = open(c_test + "/test_acc.txt", "w")
    score_total = open(c_test + "/test_score_total.txt", "w")
    


    for k in range(len(kernels)):
        score_test=[]  #tableau des score obtenus pour chaque classe pour chaque c avec le fichier de train
        score_train=[] #tableau des score obtenus pour chaque classe pour chaque c avec le fichier de test
        score_multi=[] #tableau des score obtenus pour chaque c avec un fichier contenant des exemples de chaque classe
        
        for c in l_c:
            sc_test, sc_train = testeur(c_app, c_test, c_mod, c_pred, c, k)
            score_test.append(sc_test)
            score_train.append(sc_train)
            
            sc_multi= prediction_multi_classe(c_app, c_test, c_mod, c_pred)
            #sc_multi=essai_2(c_mod)
            score_multi.append(sc_multi)
    
        
        #ecriture dans les fichiers
        
        score_train = np.array(score_train).T
        for r in score_train:
            train.write(str(r) + "\n")
        train.write("\n")
        
        score_test = np.array(score_test).T
        for r in score_test:
            test.write(str(r) + "\n")
        test.write("\n")
        
        score_total.write(str(score_multi)+"\n")
        score_total.write("\n")
    
    test.close()
    train.close()
    score_total.close()

kernels = ['linear', 'poly', 'rbf', 'sigmoïde']

val_c2 = [0.0001, 0.001, 0.1, 1, 10, 25, 50, 100]

lancement(c_app, c_test, c_mod, c_pred, val_c2, kernels)

