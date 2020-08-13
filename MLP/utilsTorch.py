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

def ventaneo_con_clase(x, fs , tamVentana,clase=None):
    lenData = x.shape[0]
    cantidad_datos_ventana=int((fs/1000)*tamVentana)
    cantidad_ventanas=int(lenData//cantidad_datos_ventana)

    nx=[] # nuevo vector
    for i in range(cantidad_ventanas):
        nx.append(np.append(x[i*cantidad_datos_ventana:i*cantidad_datos_ventana+cantidad_datos_ventana],clase))

    
    ret = np.array(nx)
    return ret, nx

def ventaneo(signal, fm, windowLenght, windowStep):
    #VENTANEO--------------------------------------------------------
    # samplesLength = math.floor((windowLenght)*fm)
    samplesLength = int((windowLenght)*fm)
    #print("SamplesLen: ", samplesLength)

    # samplesStep= math.floor((windowStep)*fm)
    samplesStep= int((windowStep)*fm)
    #print("SamplesStep: ", samplesStep)

    cantVentanas = math.ceil(signal.size/samplesStep)
    #print("CantVentanas: ", cantVentanas)

    framedSignal = []

    for i in range(0, cantVentanas):

        #Tomo una porcion de la señal de longitud samplesLength
        frame = signal[i*samplesStep:i*samplesStep+samplesLength]
        
        #Si estamos en las ultimas ventana verifico que la ventana sea del tamaño samplesLength
        #De no ser asi agrego ceros al final
        if frame.shape[0] != samplesLength:
           #print("Ventana incompleta")
            frame = np.pad(frame,(0, samplesLength - frame.shape[0]), 'constant', constant_values=0)
            #print(frame)

        #Aplico Hamming a la ventana
        frame = np.hamming(samplesLength)*frame

        #Agrego ventana a la lista de ventanas
        framedSignal.append(frame)
                
    framedSignal = np.array(framedSignal)
    return framedSignal

def remove_silences(raw_signal,  plot_result = False, treshold = 0.00025):
    print("Remove Silence")

    #Rectifico para eliminar valores negativos
    rectified_signal = np.abs(raw_signal)
    
    #Enmascaro valores menores al umbral
    x = np.ma.masked_less(rectified_signal, treshold)
    
    #Elimino valores enmascarados
    result_signal = x.compressed()
    
    if plot_result:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
        axes[0].plot(raw_signal)
        axes[1].plot(result_signal)

        plt.show()

    return result_signal


##
#   Estimate formants using LPC.
##

def get_formants(x, Fs, nformants):

    # Read from file.
    # print("leyendo archivo..")
    # Fs, x = wv.read(file_path)

    #print("Freq Mues: ", Fs)

    # Get Hamming window.
    N = len(x)
    w = hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter ([1], [1., 0.63], x1)

    # Get LPC.
    #print("calculando lpc..")
    ncoeff = 2 + (Fs / 1000)

   # print("NCOEF: ", int(ncoeff))
    A = lpc(x1, int(ncoeff))
    # print(A)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    # Fs = spf.getframerate()
    #print("calculando formantes..")
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs[:nformants]