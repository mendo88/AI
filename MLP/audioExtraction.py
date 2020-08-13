# Extraccion de caracteristicas (el que esta subido a colab)

import os
import math
import torch as tr
import torch.nn as nn
import numpy as np
import sklearn.metrics as metricas
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import librosa as lsa
from librosa import lpc
from scipy.signal import lfilter, hamming, spectrogram

from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import wavfile as wv
from utilsTorch import *

# extractor de características
def createWindowDate(directory):
    cantFormantes = 5
    cantMfcc = 13
    
    cantCaracteristicas = cantFormantes + cantMfcc
    m_features=[]
    m_clases=[]

    raw_file = np.load("./gdrive/My Drive/Colab Notebooks/databases/RAW_AUDIO_DATASET.npy", allow_pickle=True)

    # print("RAD shape: ", raw_file.shape)

    it=0
    for samplingrate, audio, clase in raw_file:      
        # print("AUDIO N° ",it,"=================================================")

        
        raw_signal = audio[0]

        #CALCULO MFCC
        features_matrix = lsa.feature.mfcc(raw_signal, samplingrate, None, cantMfcc, hop_length= int(samplingrate*(0.010)), n_fft=int(samplingrate*(0.025)))
        features_matrix=features_matrix.T #filas->ventanas y columnas->caracteristicas + clases

        #EXTRAIGO FORMANTES
        #VENTANEO
        windowLenght = 0.025 #ms
        windowStep = 0.010  #ms
        windowed_signal = ventaneo(raw_signal, samplingrate, windowLenght, windowStep)
            #Matriz que almacena las n formantes de cada ventana
        
        # CHEQUEO SI VENTANEO.SHAPE = MFCC.SHAPE
        if(features_matrix.shape[0]!=windowed_signal.shape[0]):
            # print("tam matrices distintas")
            # print("ELIMINO ULTIMA VENTANA DE FEATURES MATRIX")
            features_matrix = np.delete(features_matrix, -1, 0)


        #EXTRAIGO FORMANTES
        formantes_matrix = np.zeros((windowed_signal.shape[0], cantFormantes))

        for i in range(formantes_matrix.shape[0]):

            formantes_matrix[i] = get_formants(windowed_signal[i,:], samplingrate, cantFormantes)

        # print("FEATURES (MFCC) MATX: ", features_matrix.shape)
        # print("FORMANTES MATX: ", formantes_matrix.shape)
        
        features_matrix = np.append(features_matrix, formantes_matrix, axis = 1)

        # print("FEATURES+FORMANTES: ", features_matrix.shape ,"  <-- columnas = 13mfcc + 5form")

        #PROMEDIAR VENTANAS ------------------------------------

        #vector_averages_matrix = np.zeros((1, cantCaracteristicas))
        
        #cantVentanas = mfcc.shape[1]
        ventanasPromediadas = 7
        #Iteradores para recorrer las ventanas, f = ini + ventanas promediadas
        ini = 0
        fin = ventanasPromediadas
        contador = 0
        m_features.append(features_matrix)
        #mientras iterador de inicio menor a long de ventanas
        while(ini < features_matrix.shape[0]):

            #Calculo media de la cantidad de ventanas elegidas 
            averageVector = np.mean(features_matrix[ini:fin, :], axis = 0)
            #averageVector = np.reshape(averageVector, (1, averageVector.shape[0]))
            #m_features.append(averageVector)
            #Agrego el vector de carac promediadas a la matriz de vectores carac promediadas
            #vector_averages_matrix = np.append(vector_averages_matrix, averageVector, axis = 0)
            
            ini += ventanasPromediadas
            fin = ini + ventanasPromediadas
            contador += 1

        #Elimino la primera fila hecha de ceros que quedo por armar la matriz de ceros para hacer append despues
        #vector_averages_matrix = np.delete(vector_averages_matrix, 0, axis = 0)

        # print("FIN DE PROMEDIOS ---------------")
        # print("PROMEDIOS: ", vector_averages_matrix.shape)


        #Vector de cantFilas = cantVectoresPromediados, 1 columna, con la clase de esos vectores
        etiquetas = [clase for i in range(contador)]
        m_clases+=etiquetas
        # print("CLASES: ", etiquetas.shape)


        #AGREGO MATRIZ CON VECTORES PROMEDIADOS A MATRIZ DE FEATURES
        #m_features = np.append(m_features, vector_averages_matrix, axis = 0)
        # print("M_FEATURES Result: ", m_features.shape)
        
        #AGREGO MATRIZ CON LAS ETIQUETAS DE LOS VECTORES GENERADOS A MATRIZ DE ETIQUETAS
        #m_clases = np.append(m_clases, etiquetas, axis = 0)

        it+=1
    # np.save(save_filename_train, m_features)
    # np.save(save_filename_label, m_clases)

    # m_features: matriz de caracteristicas por fonema
    # m_clases: matriz con etiqueta de cada fonema  
    m_features=np.array(m_features)
    m_clases=np.array(m_clases)
    print("FINAL-----")
    print("M_FEATURES: ", m_features.shape)
    print("M_CLASES: ", m_clases.shape)
    return m_features, m_clases