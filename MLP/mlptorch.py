import os
import torch as tr
import torch.nn as nn
import numpy as np
import sklearn.metrics as metricas
#librosa -> 

from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import wavfile as wv
import torchvision.models as models
import torchvision.transforms as transforms



class FeaturesDataset(Dataset):
    def __init__(self, datos, etiquetas):
        self.data = np.array(np.load(datos))
        self.labels = np.array(np.load(etiquetas))
        
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):    
        dato=self.data[index]
        # print(dato)
        etiqueta=self.labels[index]
        # print(etiqueta)
        return tr.FloatTensor(dato), tr.FloatTensor(etiqueta)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.sm=nn.Softmax(1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sm(out)
        return out


if __name__=='__main__':


    #=========================================
    #   Carga de los datos
    #=========================================

    dataset = FeaturesDataset("/home/mendo/Facultad/IC/Practica/InteligenciaComputacional2019/TPFINAL/src/MLPsklearn/data_train.npy", '/home/mendo/Facultad/IC/Practica/InteligenciaComputacional2019/TPFINAL/src/MLPsklearn/train_labels.npy')
    #dataset = LangDataset("/home/mendo/Facultad/IC/Practica/InteligenciaComputacional2019/TPFINAL/src/MLPsklearn/data_train.npy")
    # dataset=np.load("./data_train.npy", allow_pickle=True)
    print()
    print("Info de datos de entrenamiento:")
    print("\tTipo del dataset: ",type(dataset.data))
    print("\tTamaño del dataset: ",len(dataset))
    print()    
    

    #=========================================
    #   entrenamiento
    #=========================================
    train_samples = int(0.9 * len(dataset)); print("\tTamaño de train: ",train_samples)
    valid_samples = len(dataset) - train_samples; print("\tTamaño de validacion: ",valid_samples)
    train, valid = random_split(dataset, [train_samples, valid_samples])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=1, shuffle=False, pin_memory=True)

    model = Net(18,10,2)
    lossfunc = tr.nn.MSELoss()
    optimizer = tr.optim.Adam(model.parameters())

    best_valid_acc = 0
    epocas = 0

    print()
    print("ENTRENAMIENTO:\n")
    while epocas < 30:

        train_loss = 0
        model.train()
        print("ENTRENANDO EPOCA: ", epocas)
        for seq, lbl in train_loader:
            optimizer.zero_grad()
            salida=model(seq)  
            label=lbl.squeeze()
            loss = lossfunc(salida, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            
        
        prediction, ground_truth = tr.LongTensor(), tr.LongTensor()
        model.eval()
        for seq, lbl in valid_loader:
            salidaRed=model(seq)
            # print("salida red: ", salidaRed.shape)
            prediction = tr.cat([prediction, tr.argmax(salidaRed, 1)])
            # print('prediction',prediction)
            # print('prediction',prediction.shape)
            ground_truth = tr.cat([ground_truth, tr.argmax(lbl,dim=1)])
            # print('ground_truth',ground_truth)
            # print('ground_truth',ground_truth.shape)
        valid_acc = metricas.accuracy_score(ground_truth.numpy(),
                                            prediction.detach().numpy())
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            epocas = 0
            tr.save(model.state_dict(), "best_model.pmt")
        else:
            epocas += 1
            
        print("Train loss: %f\t Valid acc: %f" % (train_loss, valid_acc))