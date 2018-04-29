# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script de test.
Il faut parvenir a exploiter toutes les caractéristiques des signaux EMG

Il faut préciser que la méthode de périodogramme moyenné dite de Welsh n'est pas fonctionnelle
et la fonction d'extraction des caractéristiques (extractSpecs) n'est pas terminée.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scp
import time
import csv

start = time.time()

#Récuperation des données dans le fichier CSV
def EMGAnalysis(filename1, colonnes, affiche=1):
    '''
    filename : nom du fichier que l'on veut exploiter
    colonnes : numéros des colonnes que l'on souhaite récupérer
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante 
    '''
    sigEMG = open(filename1, 'r')
    donnees = [ligne for ligne in csv.reader(sigEMG, delimiter = ';')]
    coordonnees = [] #tableau 2D contenant les données en sortie
    for number in colonnes :
        colonne = []
        for line in donnees :
            if line [0][0] != "#" : #on ignore les lignes commençant par #, ce sont des commentaires
                colonne.append(float(line[number]))
        coordonnees.append(colonne)
    sigEMG.close(); time = coordonnees[0]; data = coordonnees[1];
    
    if (affiche==1): 
        plt.figure(figsize=(16,8))
        plt.title("Signal EMG")
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude')
        plt.plot(time, data, 'b-') #affichage du signal EMG
        plt.grid()
        plt.show()
    
    return time,data #récupération des données intéressantes pour un traitement ultérieur
   
   
#------------------------------------------------------------------------------
#-------------Outils d'analyse dans le domaine temporel------------------------
#------------------------------------------------------------------------------
    
#Normalisation du signal
def Normalisation(time, data, affiche=1):
    '''
    time : échelle temporelle du signal d'entrée
    data : données d'entrée
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    #Normalisation en amplitude
    M = max(np.abs(data))
    dataNorm = [np.abs(data[i])/M for i in range(len(data))]
    
    if (affiche==1 or affiche==2):
        plt.figure(figsize=(16,16))
        #Normalisation en amplitude
        plt.subplot(211)
        plt.title("Signal EMG normalisé")
        plt.xlabel('Time [sec]'); plt.ylabel('Amplitude absolue normalisée')
        plt.plot(time, dataNorm, '-', linewidth=0.5, color='darkgreen')
        plt.grid()
        if(affiche==2):
            #EMG rectifié (valeur absolue)
            plt.subplot(212)
            plt.title("Signal EMG en valeur absolue")
            plt.xlabel('Time [sec]'); plt.ylabel('Amplitude absolue')
            plt.plot(time, np.abs(data), '-',linewidth=0.5, color='firebrick')
            plt.grid()
        plt.show()
    return dataNorm


#Filtre passe-bas numerique
def LowPassFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    order : ordre du filtre que l'on va utiliser
    cutOff : fréquence de coupure
    time : échelle temporelle du signal d'entrée
    data : données à filtrer
    feData : fréquence d'échantillonage de notre signal d'entrée
    norm : paramètre permettant de choisir si oui ou non le signal d'entrée est normalisé
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    nyquistFrequency = feData*0.5
    normCutOff = cutOff/nyquistFrequency
    b, a = scp.butter(order, normCutOff, btype='low', analog = False)
    Y_filtreBas = scp.lfilter(b,a,data)

    if (affiche==1 or affiche==2):
        
        plt.figure(figsize=(16,16))
        plt.subplot(211)
        #Reponse frequentielle du filtre
        if(norm==1) :
            w, h = scp.freqz(b, a, worN=feData)
            plt.xlabel('Fréquence normalisée'); plt.xlim(0, 0.5)
            plt.plot(w/(2*np.pi), np.abs(h), 'b')
            plt.plot(normCutOff*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff*0.5, color='gray')
        else :
            w, h = scp.freqz(b, a, worN=feData)
            plt.xlabel('Fréquence [Hz]'); plt.xlim(0, 0.5*feData)
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(cutOff, 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff, color='gray')
        
        plt.title("Lowpass Filter Frequency Response")
        plt.grid()
        
        plt.subplot(212)
        if(affiche==2):
            plt.plot(time, data, 'b-', linewidth=2, label='signal EMG')
        plt.plot(time, Y_filtreBas, 'r-', linewidth=0.5, label='Signal EMG filtre')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.grid()
        
        plt.show()
   
    return Y_filtreBas

#Filtre passe-bas numerique
def HighPassFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    order : ordre du filtre que l'on va utiliser
    cutOff : fréquence de coupure
    time : échelle temporelle du signal d'entrée
    data : données à filtrer
    feData : fréquence d'échantillonage de notre signal d'entrée
    norm : paramètre permettant de choisir si oui ou non le signal d'entrée est normalisé
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    nyquistFrequency = feData*0.5
    normCutOff = cutOff/nyquistFrequency
    b, a = scp.butter(order, normCutOff, btype='high', analog = False)
    Y_filtreHaut = scp.lfilter(b,a,data)

    if (affiche==1 or affiche==2):
        
        plt.figure(figsize=(16,16))
        plt.subplot(211)
        #Reponse frequentielle du filtre
        if(norm==1) :
            w, h = scp.freqz(b, a, worN=feData)
            plt.plot(w/(2*np.pi), np.abs(h), 'b')
            plt.plot(normCutOff*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff*0.5, color='gray')
            plt.xlabel('Frequency normalised') 
            plt.xlim(0, 0.5)
        else :
            w, h = scp.freqz(b, a, worN=feData)
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(cutOff, 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff, color='gray')
            plt.xlabel('Frequency [Hz]'); plt.xlim(0, feData)
        
        plt.title("Highpass Filter Frequency Response")
        plt.grid()
        
        plt.subplot(212)
        if(affiche==2):
            plt.plot(time, data, 'b-', linewidth=0.5, label='signal EMG')
        plt.plot(time, Y_filtreHaut, 'g-', linewidth=2, label='Signal EMG filtre')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.grid()
        
        plt.show();
   
    return Y_filtreHaut
    

#Filtre stop-bande numérique calibré pour le 50Hz
def BandStopFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    order : ordre du filtre que l'on va utiliser
    cutOff : tableau contenant les deux fréquences de coupure
    time : échelle temporelle du signal d'entrée
    data : données à filtrer
    feData : fréquence d'échantillonage de notre signal d'entrée
    norm : paramètre permettant de choisir si oui ou non le signal d'entrée est normalisé
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    nyquistFrequency = feData*0.5
    normCutOff = [cutOff[0]/nyquistFrequency, cutOff[1]/nyquistFrequency]
    b, a = scp.butter(order, normCutOff, btype='bandstop', analog = False, output='ba')
    Y_filtreCBande = scp.lfilter(b,a,data)
    
    if (affiche==1 or affiche==2):        
        plt.figure(figsize=(16,16))
        plt.subplot(211)
        #Réponse fréquentielle du filtre
        if(norm==1) :
            w, h = scp.freqz(b, a, worN=feData)
            plt.xlabel('Fréquence normalisée'); plt.xlim(0, 0.05); plt.xticks([0.005*i for i in range(11)])
            plt.plot(w/(2*np.pi), np.abs(h), linewidth=1.5, color='deepskyblue')
            plt.plot(normCutOff[0]*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff[0]*0.5, linewidth=1.5, color='black')
            plt.plot(normCutOff[1]*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff[1]*0.5, linewidth=1.5, color='black')
        else :
            w, h = scp.freqz(b, a, worN=feData)
            plt.xlabel('Fréquence [Hz]'); plt.xlim(0, 0.05*feData)
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(cutOff[0], 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff[0], color='gray')
            plt.plot(cutOff[1], 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff[1], color='gray')
       
        plt.title("Bandstop Filter Frequency Response")
        plt.grid()
        
        plt.subplot(212)
        #Affichage du signal filtré
        if (affiche==2): #dans ce cas on affiche également le signal d'origine pour comparaison
            plt.plot(time, data, 'b-', linewidth=0.5, label='signal EMG')
        plt.plot(time, Y_filtreCBande, '-',color='firebrick', linewidth=0.5, label='Signal EMG filtré')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.grid()
        
        plt.show()
    
    return Y_filtreCBande


#Moyenne du signal
def Average(time, data, ant, affiche=1):
    '''
    Moyennage
    time : échelle temporelle du signal d'entrée
    data : données à moyenner
    
    ant correspond au nombre d'echantillons que l'on va utiliser pour moyenner
    Par exemple si ant=3, l'échantillon n correspond a la moyenne des 3 échantillons
    précedents ((n-1 + n-2 + n-3)/3)
    
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    
    data_moy=[0 for i in range(ant)] #on remplace les premiers échantillons par des zéros. 
    #N'ayant pas d'échantillons antérieurs, la fonction nous renverrait une erreur autrement
    for i in range(ant,len(data)):
        data_moy.append( sum(data[i-ant:i])/ant)
    
    if (affiche==1 or affiche==2):      
        plt.figure(figsize=(16,8))
        plt.title("Signal EMG moyenné")
        plt.xlabel('Temps [sec]') 
        if (affiche==2):
            plt.plot(time, data, '-', linewidth=0.5, color='b')
        plt.plot(time, data_moy, '-', linewidth=1, color='r')
        plt.grid()
        
        plt.show()
    
    return data_moy


#Moyennage quadratique
def RMS(time, data, ant, affiche=1):
    '''
    Moyennage quadratique - RMS , root mean square
    time : échelle temporelle du signal d'entrée
    data : données à moyenner
    
    ant correspond au nombre d'echantillons que l'on va utiliser pour moyenner
    Par exemple si ant=3, l'echantillon n correspond a la moyenne des 3 echantillons
    precedents ((n-1 + n-2 + n-3)/3)
    
    affiche : code nous permettant de décider ou non de l'affichage de la courbe correspondante
    '''
    
    data_RMS=[0 for i in range(ant)] #on remplace les premiers echantillons par des zéros
    #N'ayant pas d'échantillons antérieurs, la fonction nous renverrait une erreur autrement
    for i in range(ant,len(data)):
        data_RMS.append( np.sqrt(sum(np.square(data[i-ant:i])))/ant) #liste des données RMS
    
    if (affiche==1 or affiche==2):
        plt.figure(figsize=(16,8))
        plt.title("Signal EMG en moyenne quadratique")
        plt.xlabel('Temps [sec]') 
        if (affiche==2):
            plt.plot(time, data, linewidth=0.5, color='b')
        plt.plot(time, data_RMS, linewidth=1, color='r')
        plt.grid()
        
        plt.show()
    
    return data_RMS


#Seuil et sélection des données intéressantes du signal
#Fonction à n'utiliser que si le signal est normalisé

def SelectionData(time, data, feData, seuil, affiche=1):
    '''
    time : échelle temporelle du signal d'entrée
    data : données dont on souhaite sélectionner les parties intéressantes
    seuil : valeur permettant de définir à partir de quelle amplitude les valeurs du signal sont intéressantes
    
    Cette fonction renvoie une matrice dont les lignes contiennent les différents 
    moments du signal où les données étaient intéressantes à traiter, synonyme de 
    "beaucoup de mouvement" ou bien de contraction musculaire
    '''
    timeInt=[]
    dataInt=[]
    matriceData=[]
    matriceTime=[]
    
    premiersDepassements = []
    i = 0                       #variable de boucle while
    
    while i <= len(data)-1 :
        if data[i] >= seuil :
            # première intervalle de temps sur laquelle on souhaite capturer
            deltaTemps = feData/6
            #récuperation des instants ou le seuil est dépassé pour la première fois
            premiersDepassements.append(time[i])
            #démarrage du timer de capture
            j = 0
            while j <= int(deltaTemps) :
                #juste avant la fin de la capture, on vérifie qu'il n'y a plus de signal intéressant après
                if j == int(deltaTemps)-1 :
                    liste = []
                    for k in range(i, i+40):
                        liste.append(data[k])
                    #on regarde le max sur 50 échantillons après le dernier échantillon capturé
                    if max(liste) >= seuil :
                        #on prolonge le temps de capture si la condition est verifiée
                        deltaTemps += feData/7
                #on ajoute dans nos liste de données les parties de signal capturees
                dataInt.append(data[i])
                timeInt.append(time[i])
                i += 1
                j += 1
            
            matriceData.append(dataInt)
            matriceTime.append(timeInt)
            deltaTemps = feData/3
            timeInt=[]
            dataInt=[]
        i += 1
            
    if (affiche==1):
        for l in range(len(matriceTime)):
            plt.figure(figsize=(16,8))
            plt.title("Partie du signal intéressante n°"+str(l+1))
            plt.xlabel("Temps [sec]")
            plt.ylabel("Amplitude")
            plt.plot(matriceTime[l], matriceData[l], linewidth=1.5, color='r')
            plt.grid()
            plt.show()
            
    return matriceTime, matriceData, premiersDepassements




#------------------------------------------------------------------------------
#-------------------Outils d'analyse frequentielle-----------------------------
#------------------------------------------------------------------------------

def FourierTransform(time, data, feData, norm=1, affiche=1):
    '''
    time : échelle temporelle du signal d'entrée
    data : données dont on souhaite calculer la transformee de Fourier
    feData : fréquence d'échantillonage de notre signal d'entrée
    norm : valant 0 ou 1, ce paramètre permet de faire cette transformée en fréquence normalisée ou non
    '''
    
    #Calcul de la transformeé de Fourier
    fftEMG = np.fft.fft(data)
    freq_norm = np.fft.fftfreq(len(time))
    freq = [freq_norm[i]*feData for i in range(len(freq_norm))]
    
    if (affiche==1):
        plt.figure(figsize=(16,16))
        
        if (norm==1):
            #Transformee de Fourier en frequence normalisée
    
            #Partie réelle
            plt.subplot(211)
            plt.title('Fourier Transform')
            plt.xlabel('Fréquence normalisée'); plt.xlim([0,0.5])
            plt.ylabel('Partie réelle');plt.ylim([-30,30])
            plt.plot(freq_norm, np.real(fftEMG), '-', color='royalblue')
            plt.grid()
            #Partie imaginaire
            plt.subplot(212)
            plt.title('Fourier Transform')
            plt.xlabel('Fréquence normalisée'); plt.xlim([0,0.5])
            plt.ylabel('Partie imaginaire');plt.ylim([-30,30])
            plt.plot(freq_norm, np.imag(fftEMG), '-', color='darkorange')
            plt.grid()
        else :
            #Transformée de Fourier en frequence denormalisée
        
            #Partie réelle
            plt.subplot(211)
            plt.title('Fourier Transform')
            plt.xlabel('Fréquence');plt.xlim([0,feData])
            plt.ylabel('Partie réelle');plt.ylim([-50,50])
            plt.plot(freq, np.real(fftEMG), '-', color='royalblue')
            plt.grid()
        
            #Partie imaginaire
            plt.subplot(212)
            plt.title('Fourier Transform')
            plt.xlabel('Fréquence'); plt.xlim([0,feData])
            plt.ylabel('Partie imaginaire');plt.ylim([-50,50])
            plt.plot(freq, np.imag(fftEMG), '-', color='darkorange')
            plt.grid()
        
        plt.show()
    
    #choix de la fréquence de sortie si l'on a travaillé en normalisé ou non
    if(norm==1):
        freq_out = freq_norm
    else:
        freq_out = freq
    
    return freq_out,fftEMG

def PowerSpectralDensity(frequence, data, method=1, norm=1, affiche=1):
    '''
    fréquence : echelle fréquentielle du signal d'entrée
    data : données issues d'une transformée de Fourier dont on souhaite calculer la DSP
    method : périodogramme ou périodogramme moyenné
    norm : valant 0 ou 1, ce paramètre permet de faire un affichage en fréquence normalisée ou non
    '''
    #Périodogramme
    if (method==1):
        
        PSD = np.square(np.abs(data))
       
        if(affiche==1): 
            plt.figure(figsize=(16,8))
            plt.title('Densité spectrale de puissance (méthode Périodogramme)')
            plt.xlabel('Fréquence')
            plt.ylim(1e-3,1e5); plt.yscale('log')
            if(norm==1):
                plt.xlim([0,0.5]); plt.xticks([0.05*i for i in range(11)])
            else :
                plt.xlim([-100,100])
            plt.semilogy(frequence, PSD, '-', linewidth=1.3, color='limegreen')
            plt.grid(True, which='both', ls='-', color='0.65')
            plt.show()
    
    #Périodogramme moyenné - Methode de Welch
    if (method==2):
        
        nb_element = 100                            #nombre d'élements par intervalle
        nb_intervalle = len(data) // nb_element     #nombre d'intervalle de nb_element dans mon signal
        print(nb_intervalle)
        PSD_moy=[]
        freq_moy=[]
        for i in range(0,nb_intervalle):
            min = i*nb_element                      #Calcul de l'index min et max de chaque intervalle
            max = min + nb_element
            #Calcul du PSD ainsi que de la fréquence moyenne sur l'intervalle
            PSD_intervalle = np.square(np.abs(data[min : max]))
            frequence_intervalle = np.mean(frequence[min:max])
            #On ajoute la fréquence et le PSD calculés sur un intervalle à une liste
            PSD_moy.append(PSD_intervalle)
            freq_moy.append(frequence_intervalle)
        
        PSD = PSD_moy
        
        if(affiche==1):           
            plt.figure(figsize=(16,8))
            plt.title('Densité spectrale de Puissance (méthode Périodogramme moyenné)')
            plt.xlabel('Fréquence')
            plt.ylim(1e-3,1e5); plt.yscale('log')
            if(norm==1):
                plt.xlim([0,0.5]); plt.xticks([0.05*i for i in range(11)])
            else :
                plt.xlim([-100,100])
            plt.semilogy(freq_moy, PSD, '-', linewidth=1.3, color='darkgreen')
            plt.grid(True, which='both', ls='-', color='0.65')
            plt.show()

    return PSD

def FondamentalLocal(time,data):
    '''
    Cette fonction a pour but de recuperer les parties du signal ou la frequence es elevee
    Pour se faire, on calcule la transformee de Fourier sur de petits intervalles, et on
    regarde la valeur de la frequence fondamentale. Si cette derni`re depasse un certain seuil,
    on conserve le signal.
    '''    
    matriceData=[]
    matriceTime=[]
    
    while i <= len(data)-1:
        for j in range(1,100):
            i += 1
    #modification en cours
    return matriceTime, matriceData



#------------------------------------------------------------------------------
#-------------------Extraction de caracteristiques-----------------------------
#------------------------------------------------------------------------------

def ExtractSpecs(time, data) : 
    #Recuperation de la moyenne du signal
    moyenne = sum(data)/len(data)
    maximum = max(data)
    minimum = min(data)
    sommeCarre = 0
    for i in range(len(data)):
        sommeCarre += pow(data[i],2)
    valEfficace = np.sqrt(sommeCarre/len(data))
    
    return moyenne, maximum, minimum, valEfficace














#------------------------------------------------------------------------------
#-----------------------Éxécution de la fonction-------------------------------
#------------------------------------------------------------------------------

'''
SigEMG.csv
On a 50 860 echantillons sur 12.715 secondes
La frequence d'echantillonage de notre signal est donc : 4000Hz

Premiers tests de Maxime : Test1.csv
La frequence d'echantillonage de notre signal est : 1280Hz

'''
feData=1280
tps,data = EMGAnalysis("Test1.csv",[0,1],affiche=1)
#------------------------------------------------------------------------------
#-----------------------------Dénormalisé--------------------------------------
#------------------------------------------------------------------------------
dataFiltr = BandStopFilter(5, [47,53], tps, data, feData, norm=0, affiche=0)

#dataMoy = Average(tps, data, 5, affiche=0)
#dataRMS = RMS(tps, data, 5, affiche=0)

freq1, fftEMG = FourierTransform(tps, data, feData, norm=0, affiche=0)
freq2, fftEMGFiltr = FourierTransform(tps, dataFiltr, feData, norm=0, affiche=0)

PowerSpectralDensity(freq1,fftEMG, method=1, norm=0, affiche=0)
PowerSpectralDensity(freq2,fftEMGFiltr, method=1, norm=0, affiche=0)

#LowPassFilter(25, 400, tps, data, feData, norm=0, affiche=0)
#HighPassFilter(10, 10, tps, dataNorm, feData, norm=0, affiche=0)


#------------------------------------------------------------------------------
#-------------------------------Normalisé--------------------------------------
#------------------------------------------------------------------------------
dataNorm = Normalisation(tps, data, affiche=1)
dataFiltrNorm = BandStopFilter(5,[47,53], tps, dataNorm, feData, norm=1, affiche=0)

matTime, matData, premDepass = SelectionData(tps, dataNorm, feData, 0.2, affiche=1)
print(premDepass)

#dataMoyNorm = Average(tps, dataNorm, 5, affiche=2)
#dataRMSNorm = RMS(tps, dataNorm, 5, affiche=2)

freq1Norm, fftEMGNorm = FourierTransform(tps, dataNorm, feData, norm=1, affiche=1)
freq2Norm, fftEMGFiltrNorm = FourierTransform(tps, dataFiltrNorm, feData, norm=1, affiche=0)

PowerSpectralDensity(freq1Norm,fftEMGNorm, method=1, norm=1, affiche=1)
PowerSpectralDensity(freq2Norm,fftEMGFiltrNorm, method=1, norm=1, affiche=1)

#LowPassFilter(25, 400, tps, dataNorm, feData, norm=1, affiche=0)
#HighPassFilter(10, 10, tps, dataNorm, feData, norm=1, affiche=0)







#Affichage du temps necessaire à la réalisation de toutes les opérations de traitement
#afin de savoir si l'utilisation en temps réel est possible
print("temps nécessaire = %.5e" %(time.time() - start))










