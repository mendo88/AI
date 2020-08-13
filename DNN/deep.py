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


class LanguageDNN(nn.Module):
    def __init__(self):
        super(LanguageDNN, self).__init__()

        self.conv_linear_match=512
        #agregar pading ambos lados
        self.conv_layers = nn.Sequential(

            nn.BatchNorm1d(1),
            nn.Conv1d(1,64, 3, 1,1),
            nn.ReLU(),

            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64,3,1,1),
            nn.ReLU(),

            nn.MaxPool1d(8),

            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128,3,1,1),
            nn.ReLU(),
            
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128,3,1,1),
            nn.ReLU(),

            nn.MaxPool1d(8),
            
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256,3,1,1),
            nn.ReLU(),
            
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256,3,1,1),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Conv1d(256, self.conv_linear_match,3,1,1),
            nn.LogSigmoid(),
            nn.AdaptiveMaxPool1d(1))
        
        self.linear_layers = nn.Sequential(
            # nn.BatchNorm1d(self.conv_linear_match),
            nn.Dropout(0.2),
            nn.Linear(self.conv_linear_match, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7),
            nn.Softmin(1)
            )
                
    def forward(self, x):

        x = self.conv_layers(x)
        # print(x.shape)
        x = x.view(-1, self.conv_linear_match) 
        # print(x.shape)
        x = self.linear_layers(x)
        # print(x.shape)
        return x

class LanguageCNN(nn.Module):
    def __init__(self):
        super(LanguageCNN, self).__init__()

        self.conv_linear_match=64
        #agregar pading ambos lados
        self.conv_layers = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 4, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.AvgPool1d(4),
            nn.Conv1d(4, 8, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(4),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(4),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.AvgPool1d(4),
            nn.Conv1d(48, self.conv_linear_match, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))

        
        
        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(self.conv_linear_match),
            #nn.Dropout(0.2),
            nn.Linear(self.conv_linear_match, 32),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(32, 8),
            )
                
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 1, x.shape[1])
        # print(x.shape)
        # print(x)
        x = self.conv_layers(x)
        # print(x.shape)
        x = x.view(-1, self.conv_linear_match) 
        # print(x.shape)
        x = self.linear_layers(x)
        # print(x.shape)
        return x

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# esta clase está hecha para otorgar datos que no tienen la misma longitud, por eso, se utiliza un mapa
# para poder guardar ahí los datos. Todos los datos estan en un map {}
class LangMapaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.load(data, allow_pickle=True).item()
        self.labels = np.load(labels, allow_pickle=True).item()
        
    def __len__(self):
        return(len(self.labels.keys()))

    def __getitem__(self, index):    
        dato=self.data[int(index)]
        etiqueta=self.labels[int(index)]
        return tr.Tensor(dato), tr.FloatTensor(etiqueta)

#   esta clase maneja arreglos de numpy comunes
class LangDataset(Dataset):
    def __init__(self, file, transform=None):
        self.data = np.array(np.load(file))
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):    

        dato=self.data[index,:-8]
        if self.transform:
            dato = transform(dato)
        # print(dato)
        etiqueta=self.data[index,-8:]
        # print(etiqueta)
        return tr.FloatTensor(dato), tr.FloatTensor(etiqueta)

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

if __name__=='__main__':


    #=========================================
    #   Carga de los datos
    #=========================================

    # transform = transforms.Compose(
    #     [
    #         ta.transforms.ToTensor(),
    #         ta.transforms.Spectogram()
    #     ]
    # )

    # dataset = LangMapaDataset("./spectogram_data_train.npy", './spectogram_labels_train.npy')
    # dataset = LangDataset("./data_train.npy")
    dataset = FeaturesDataset("./features_data_dataset.npy", './features_labels_dataset.npy')
    # dataset=np.load("./data_train.npy", allow_pickle=True)
    print()
    print("Info de datos de entrenamiento:")
    print("\tTipo del dataset: \t",type(dataset.data))
    print("\tTamaño del dataset: \t",len(dataset))
    

    

    #=========================================
    #   entrenamiento
    #=========================================
    train_samples = int(0.9 * len(dataset)); print("\tTamaño de train: \t",train_samples)
    valid_samples = len(dataset) - train_samples; print("\tTamaño de validacion: \t",valid_samples)

    print()
    print("Ejemplo de un dato:")
    print(dataset.__getitem__(12))    
    print()
    print()

    train, valid = random_split(dataset, [train_samples, valid_samples])
    batch_size=12000
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = LanguageDNN().cuda()
    # model = Net(39, 50, 8).cuda()
    # lossfunc = tr.nn.MSELoss()
    lossfunc = tr.nn.CrossEntropyLoss()
    optimizer = tr.optim.Adam(model.parameters())

    best_valid_acc = 0
    epocas = 0

    print()
    print("ENTRENAMIENTO:\n")
    while epocas < 100:

        train_loss = 0
        model.train()
        print("ENTRENANDO EPOCA: ", epocas)
        for seq, lbl in train_loader:
            seq, lbl = seq.cuda(), lbl.cuda()
            optimizer.zero_grad()
            salida=model(seq)  
            label=lbl.squeeze()
            loss = lossfunc(salida, tr.argmax(label,1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            
        
        prediction, ground_truth = tr.LongTensor(), tr.LongTensor()
        model.eval()
        for seq, lbl in valid_loader:
            seq, lbl = seq.cuda(), lbl.cuda()
            salidaRed=model(seq)
            # print("salida red: ", salidaRed.shape)
            prediction = tr.cat([prediction, tr.argmax(salidaRed, 1).cpu()])
            # print('prediction',prediction)
            # print('prediction',prediction.shape)
            ground_truth = tr.cat([ground_truth, tr.argmax(lbl,dim=1).cpu() ])
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
            
        print("Train loss: %f\t Valid acc: %f\t Best acc: %f" % (train_loss, valid_acc, best_valid_acc))
    
