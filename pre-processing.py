# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script de test.
Il faut parvenir a exploiter toutes les caracteristiques des signaux EMG
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scp
import time
import csv

start = time.time()

#------------------------------------------------------------------------------
#-------------Outils d'analyse dans le domaine temporel------------------------
#------------------------------------------------------------------------------

#Recuperation des donnees dans le fichier csv
def EMGAnalysis(filename1, colonnes, affiche=1):
    '''
    filiename : nom du fichier que l'on veut exploiter
    colonnes : numeros de colonnes a recuperer
    '''
    sigEMG = open(filename1, 'r')
    donnees = [ligne for ligne in csv.reader(sigEMG, delimiter = ';')]
    coordonnees = []
    for number in colonnes :
        colonne = []
        for line in donnees :
            if line [0][0] != "#" :
                colonne.append(float(line[number]))
        coordonnees.append(colonne)
    sigEMG.close(); time = coordonnees[0]; data = coordonnees[1];
    
    if (affiche==1): 
        plt.figure(figsize=(16,8))
        #Affichage du signal EMG
        plt.title("Signal EMG")
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude'); plt.ylim([-0.9,0.9]);
        plt.plot(time, data, 'k-')
        plt.grid()
    
    return time,data
    
    
#Normalisation du signal
def Normalisation(time, data, affiche=1):
    #Normalisation en amplitude
    M = max(data)
    dataNorm = [np.abs(data[i])/M for i in range(len(data))]
    
    if (affiche==1 or affiche==2):
        plt.figure(figsize=(16,16))
        #Normalisation en amplitude
        plt.subplot(211)
        plt.title("Signal EMG normalise")
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude absolue normalisee')
        plt.plot(time, dataNorm, '-', linewidth=0.5, color='salmon')
        if(affiche==2):
            #EMG rectifie (valeur absolue)
            plt.subplot(212)
            plt.title("Signal EMG en valeur absolue")
            plt.xlabel('Time [sec]')
            plt.ylabel('Amplitude absolue')#; plt.ylim([-0.1,0.9]);
            plt.plot(time, np.abs(data), '-', color='firebrick')
    return dataNorm


#Filtre passe-bas numerique
def LowPassFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    cutOff : frequence de coupure
    time : echelle temporelle de notre signal
    data : donnees a filtrer
    feData : frequence d'echantillonage de notre signal de depart, data
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
            plt.xlabel('Frequency [Hz]'); plt.xlim(0, 0.5*feData)
        
        plt.title("Lowpass Filter Frequency Response")
        plt.grid()
        
        plt.subplot(212)
        if(affiche==2):
            plt.plot(time, data, 'b-', linewidth=2, label='signal EMG')
        plt.plot(time, Y_filtreBas, 'r-', linewidth=0.5, label='Signal EMG filtre')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.grid()
        
        plt.show();
   
    return Y_filtreBas

#Filtre passe-bas numerique
def HighPassFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    cutOff : frequence de coupure
    time : echelle temporelle de notre signal
    data : donnees a filtrer
    feData : frequence d'echantillonage de notre signal de depart, data
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
    

#Filtre stop-bande numerique calibre pour le 50Hz
def BandStopFilter(order, cutOff, time, data, feData, norm=1, affiche=1):
    '''
    cutOff ; tableau contenant les deux frequences de coupure
    time : echelle temporelle de notre signal
    data : donnees a filtrer
    feData : frequence d'echantillonage de notre signal de depart, data
    '''
    nyquistFrequency = feData*0.5
    normCutOff = [cutOff[0]/nyquistFrequency, cutOff[1]/nyquistFrequency]
    b, a = scp.butter(order, normCutOff, btype='bandstop', analog = False, output='ba')
    Y_filtreCBande = scp.lfilter(b,a,data)
    
    if (affiche==1 or affiche==2):
        
        plt.figure(figsize=(16,16))
        plt.subplot(211)
        #Reponse frequentielle du filtre
        if(norm==1) :
            w, h = scp.freqz(b, a, worN=feData)
            plt.plot(w/(2*np.pi), np.abs(h), 'b')
            plt.plot(normCutOff[0]*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff[0]*0.5, color='gray')
            plt.plot(normCutOff[1]*0.5, 1/np.sqrt(2), 'ko')
            plt.axvline(normCutOff[1]*0.5, color='gray')
            plt.xlabel('Frequency normalised') 
            plt.xlim(0, 0.05)
        else :
            w, h = scp.freqz(b, a, worN=feData)
            plt.plot(feData*w/(2*np.pi), np.abs(h), 'b')
            plt.plot(cutOff[0], 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff[0], color='gray')
            plt.plot(cutOff[1], 1/np.sqrt(2), 'ko')
            plt.axvline(cutOff[1], color='gray')
            plt.xlabel('Frequency [Hz]'); plt.xlim(0, 0.05*feData)
       
        plt.title("Bandstop Filter Frequency Response")
        plt.grid()
        
        plt.subplot(212)
        #Affichage du signal filtre
        if (affiche==2):
            plt.plot(time, data, 'r-', linewidth=2, label='signal EMG')
        plt.plot(time, Y_filtreCBande, 'b-', linewidth=1, label='Signal EMG filtre')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.grid()
        
        plt.show();
    
    return Y_filtreCBande


#Moyenne du signal
def Average(time, data, ant, affiche=1):
    '''
    moyennage
    
    ant correspond au nombre d'echantillons que l'on va utiliser pour moyenner
    Par exemple si ant=3, l'echantillon n correspond a la moyenne des 3 echantillons
    precedents ((n-1 + n-2 + n-3)/3)
    '''
    
    data_moy=[0 for i in range(ant)] #on remplace les premiers echantillons par des zeros
    for i in range(ant,len(data)):
        data_moy.append( sum(data[i-ant:i])/ant)
    
    if (affiche==1):
        
        plt.figure(figsize=(16,8))
        if (affiche==2):
            plt.plot(time, data, 'b')
        plt.plot(time, data_moy, 'r')
        plt.title("Signal EMG moyenne")
        plt.xlabel('Temps [sec]') 
        plt.grid()
        
        plt.show();
    
    return data_moy


#Moyennage quadratique
def RMS(time, data, ant, affiche=1):
    '''
    moyennage quadratique - RMS , root mean square
    
    ant correspond au nombre d'echantillons que l'on va utiliser pour moyenner
    Par exemple si ant=3, l'echantillon n correspond a la moyenne des 3 echantillons
    precedents ((n-1 + n-2 + n-3)/3)
    '''
    
    data_RMS=[0 for i in range(ant)] #on remplace les premiers echantillons par des zeros
    for i in range(ant,len(data)):
        data_RMS.append( np.sqrt(sum(np.square(data[i-ant:i])))/ant) #liste des donnees RMS
    
    if (affiche==1 or affiche==2):
        
        plt.figure(figsize=(16,8))
        if (affiche==2):
            plt.plot(time, data, 'b')
        plt.plot(time, data_RMS, 'r')
        plt.title("Signal EMG moyenne quadritique")
        plt.xlabel('Temps [sec]') 
        plt.grid()
        
        plt.show();
    
    return data_RMS


#Seuil et selection des donnees interessantes du signal
#Fonction a n'utiliser que si le signal est normalise

def SelectionData(time, data, feData, seuil, affiche=1):
    '''
    time : echelle temporelle de notre signal
    data : donnees dont on souhaite calculer la transformee de Fourier
    seuil : valeur permettant de definir les valeurs du signal interessantes
    
    Cette fonction renvoie une matrice dont les lignes contiennent les differents 
    moments ou les donnees etaient interessantes a traiter 
    "beaucoup de mouvement"
    '''
    timeInt=[]
    dataInt=[]
    matriceData=[]
    matriceTime=[]
    
    premiersDepassements = []
    i = 0                       #variable de boucle while
    deltaTemps = feData/3       #intervalle de temps sur laquelle on souhaite capturer
    
    while i <= len(data)-1 :
        if data[i] >= seuil :
            #recuperation des instants ou le seuil est depasse pour la premiere fois
            premiersDepassements.append(time[i])
            #demarrage du timer
            j = 0
            while j <= int(deltaTemps) :
                #juste avant la fin de la capture, on verifie qu'il n'y a plus de signal interessant apres
                if j == int(deltaTemps)-1 :
                    for k in range(i, i+30):
                        liste = []
                        liste.append(data[k])
                    if max(liste) >= seuil :
                        #on prolonge le temps de capture si la condition est verifiee
                        deltaTemps += feData/4
                #on ajoute dans nos liste de donnees les parties de signal capturees
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
            plt.figure(figsize=(12,8))
            plt.title("Partie du signal interessante numero : "+str(l+1))
            plt.xlabel("Temps [sec]")
            plt.ylabel("Amplitude")
            plt.plot(matriceTime[l], matriceData[l], 'r')
            plt.grid()
        plt.show();
            
    return matriceTime, matriceData, premiersDepassements




#------------------------------------------------------------------------------
#-------------------Outils d'analyse frequentielle-----------------------------
#------------------------------------------------------------------------------

def FourierTransform(time, data, feData, norm=1, affiche=1):
    '''
    time : echelle temporelle de notre signal
    data : donnees dont on souhaite calculer la transformee de Fourier
    feData : frequence d'echantillonage de notre signal de depart, data
    norm : valant 0 ou 1, ce parametre permet de faire cette transformee en frequence normalisee ou non
    '''
    
    #Calcul de la transformee de Fourier
    fftEMG = np.fft.fft(data)
    freq_norm = np.fft.fftfreq(len(time))
    freq = [freq_norm[i]*feData for i in range(len(freq_norm))]
    
    if (affiche==1):
        
        plt.figure(figsize=(16,16))
        if (norm==1):
            #Transformee de Fourier en frequence normalisee
    
            #Partie reelle
            plt.subplot(211)
            plt.xlabel('Frequence normalisee'); plt.xlim([0,0.5])
            plt.ylabel('Partie reelle');plt.ylim([-50,50])
            plt.title('Fourier Transform');
            plt.title('Fourier Transform');
            plt.plot(freq_norm, np.real(fftEMG), '-', color='royalblue')
            #Partie imaginaire
            plt.subplot(212)
            plt.xlabel('Frequence normalisee'); plt.xlim([0,0.5])
            plt.ylabel('Partie imaginaire');plt.ylim([-50,50])
            plt.title('Fourier Transform');
            plt.title('Fourier Transform');
            plt.plot(freq_norm, np.imag(fftEMG), '-', color='darkorange')
        else :
            #Transformee de Fourier en frequence denormalisee
        
            #Partie reelle
            plt.subplot(211)
            plt.xlabel('Frequence');plt.xlim([0,250])
            plt.ylabel('Partie reelle');plt.ylim([-50,50])
            plt.title('Fourier Transform');
            plt.title('Fourier Transform');
            plt.plot(freq, np.real(fftEMG), '-', color='royalblue')
        
            #Partie imaginaire
            plt.subplot(212)
            plt.xlabel('Frequence'); plt.xlim([0,250])
            plt.ylabel('Partie imaginaire');plt.ylim([-50,50])
            plt.title('Fourier Transform');
            plt.plot(freq, np.imag(fftEMG), '-', color='darkorange')
        
        plt.show();
    
    if(norm==1):
        freq_out = freq_norm
    else:
        freq_out = freq
    
    return freq_out,fftEMG

def PowerSpectralDensity(frequence, data, method=1, norm=1, affiche=1):
    '''
    frequence : echelle frequentielle de notre signal
    data : donnees issues d'une transformee de Fourier dont on souhaite calculer la DSP
    method : periodogramme ou periodogramme moyenne
    norm : valant 0 ou 1, ce parametre permet de faire un affichage en frequence normalisee ou non
    '''
    #Periodogramme
    if (method==1):
        
        PSD = np.square(np.abs(data))
       
        if(affiche==1): 
            plt.figure(figsize=(16,8))
            plt.xlabel('Frequence')
            if(norm==1):
                plt.xlim([-0.1,0.1])
            else :
                plt.xlim([-100,100])
            plt.title('Densite spectrale de Puissance (methode Periodgramme)')
            plt.semilogy(frequence, PSD, '-', color='darkgreen')
            plt.grid()
    
    #Periodogramme moyenne - Methode de Welch
    if (method==2):
        
        nb_element = 100                            #nombre d'elements par intervalle
        nb_intervalle = len(data) // nb_element     #nombre d'intervalle de nb_element dans mon signal
        print(nb_intervalle)
        PSD_moy=[]
        freq_moy=[]
        for i in range(0,nb_intervalle):
            min = i*nb_element                      #Calcul de l'index min et max de chaque intervalle
            max = min + nb_element
            #Calcul du PSD ainsi qe de la frequence moyenne sur l'intervalle
            PSD_intervalle = np.square(np.abs(data[min : max]))
            frequence_intervalle = np.mean(frequence[min:max])
            #On ajoute la frequence et le PSD calcule sur un intervalle a une liste
            PSD_moy.append(PSD_intervalle)
            freq_moy.append(frequence_intervalle)
        
        PSD = PSD_moy
       
        if(affiche==1):           
            plt.figure(figsize=(16,8))
            plt.xlabel('Frequence');plt.xlim([-200,200])
            if(norm==1):
                plt.xlim([-0.1,0.1])
            else :
                plt.xlim([-100,100])
            plt.title('Densite spectrale de Puissance (methode Periodgramme moyenne)');
            plt.semilogy(freq_moy, PSD, '-', color='darkred')
            plt.grid()

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
    
    
    return moyenne, maximum,minimum, valEfficace
    

#Observation et zoom sur des graphes
def Trace(abscisse, ordonnee, xMin, xMax, yMin, yMax):
     plt.title('Test et Zoom sur le signal')
     plt.xlim([xMin,xMax])
     plt.ylim([yMin,yMax])
     plt.plot(abscisse, ordonnee, '-', linewidth=1, color='black')


#------------------------------------------------------------------------------
#-----------------------Execution de la fonction-------------------------------
#------------------------------------------------------------------------------

'''
SigEMG.csv
On a 50 860 echantillons sur 12.715 secondes
La frequence d'echantillonage de notre signal est donc : 4000Hz

Premiers tests de Maxime : Test1.csv
La frequence d'echantillonage de notre signal est : 1280Hz

'''
feData=1280
#--------------------Denormalise--------------------
tps,data = EMGAnalysis("Test1.csv",[0,1],affiche=1)

#LowPassFilter(10, 100, tps, data, feData, norm=1, affiche=0)
dataFiltr = BandStopFilter(3, [45,55], tps, data, feData, norm=0, affiche=0)
Average(tps, data, 5, affiche=0)
RMS(tps, data, 5, affiche=0)

freq1, fftEMG = FourierTransform(tps, data, feData, norm=0, affiche=0)
freq2, fftEMGFiltr = FourierTransform(tps, dataFiltr, feData, norm=0, affiche=0)
PowerSpectralDensity(freq1,fftEMG, method=1, norm=0, affiche=0)
PowerSpectralDensity(freq2,fftEMGFiltr, method=1, norm=0, affiche=0)



#--------------------Normalise--------------------
dataNorm = Normalisation(tps, data, affiche=1)
matTime, matData, premDepass = SelectionData(tps, dataNorm, feData, 0.2, affiche=1)
print(premDepass)

LowPassFilter(25, 400, tps, dataNorm, feData, norm=1, affiche=0)
dataFiltrNorm = BandStopFilter(3, [45,55], tps, dataNorm, feData, norm=1, affiche=0)

freq1, fftEMGNorm = FourierTransform(tps, dataNorm, feData, norm=1, affiche=0)
FourierTransform(tps, dataNorm, feData, norm=1, affiche=0)
freq2, fftEMGFiltrNorm = FourierTransform(tps, dataFiltrNorm, feData, norm=1, affiche=0)
PowerSpectralDensity(freq1,fftEMGNorm, method=1, norm=1, affiche=0)
PowerSpectralDensity(freq2,fftEMGFiltrNorm, method=1, norm=1, affiche=0)



#Affichage du temps necessaire a la realisation de toutes les operations de traitement
#afin de savoir si l'utilisation en temps reel est possible
print("temps nécessaire = %.5e" %(time.time() - start))




#--------------------Nouveaux tests--------------------

#Recuperation de l'enveloppe du signal selon differentes frequences de coupure
#for i in [5,10,40]:
#    LowPassFilter(4, i, tps, dataNorm, feData, norm=0, affiche=0)










