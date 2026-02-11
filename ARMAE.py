import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader, random_split

from AutoEncoder import *

import copy
import time
import os

import tqdm
from ast import literal_eval


class ARMAE:
    def __init__(self, dataSize,  learningRate=1e-3, maxEpoch=100,
                 batchSize=128,  hiddenSize='dataSize',  likeness=0.4,minSupp=0,minConf=0,columns=[],isLoadedModel=False, IM=['support','confidence'],):
        self.dataSize = dataSize
        self.learningRate = learningRate
        self.likeness = likeness
        self.minConf = minConf
        self.minSupp = minSupp
        self.IM=IM
        self.hiddenSize = hiddenSize
        self.isLoadedModel = isLoadedModel
        self.columns = columns
        if self.hiddenSize == 'dataSize':
            self.hiddenSize = self.dataSize
        self.maxEpoch = maxEpoch
        self.x = []
        self.y_ = []
        self.batchSize = batchSize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoEncoder(self.dataSize).to(self.device)
        self.criterion = L1Loss()
        self.optimizer = torch.optim.Adam(
        self.model.parameters(), lr=self.learningRate)

        self.results = []


    def dataPretraitement(self, d, val_split=0.2):
        self.columns = d.columns
        fullTensor = torch.tensor(d.values).float()

        dataset_size = len(fullTensor)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(fullTensor, [train_size, val_size])

        trainLoader = DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True)
        valLoader = DataLoader(val_dataset, batch_size=self.batchSize, shuffle=False)

        return trainLoader, valLoader

    def save(self, p):
        self.model.save(p)

    def load(self, encoderPath,decoderPath):
        self.model.load(encoderPath,decoderPath)
    
    def train(self, trainLoader, valLoader, modelPath, patience=10, min_delta=1e-3, warmUpEpochs=5):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.maxEpoch):

            # -------- TRAIN --------
            self.model.train()
            train_loss = 0

            for data in trainLoader:
                d = data.to(self.device)

                output = self.model.forward(d)
                loss = self.criterion(output[0], d)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()

            train_loss /= len(trainLoader)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for data in valLoader:
                    d = data.to(self.device)
                    output = self.model(d)
                    loss = self.criterion(output[0], d)
                    val_loss += loss.item()

            val_loss /= len(valLoader)

            print(f"Epoch [{epoch+1}/{self.maxEpoch}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # -------- EARLY STOPPING --------
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0

                if not self.isLoadedModel:
                    self.save(os.path.join(modelPath, "best_model"))

            else:
                if epoch < warmUpEpochs:
                    continue  # Skip early stopping during warm-up epochs
                
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return epoch




    def computeMeasures(self, antecedent, consequent,data):
        measures = []
        if 'support' in self.IM:
            rules = copy.deepcopy(antecedent)
            rules.append(consequent)
            PAC = data[data.columns[rules]]
            PAC = np.sum(PAC,axis=1)
            PAC = PAC==len(rules)
            PAC = np.sum(PAC)
            PAC = PAC/len(data)
            measures.append(round(PAC,2))
        if 'confidence' in self.IM:
            PA = data[data.columns[antecedent]]
            PA = np.sum(PA, axis=1)
            PA = PA == len(antecedent)
            PA = np.sum(PA)
            PA = PA / len(data)
            if PA !=0:
                conf = PAC/PA
            else:
                conf = 0
            measures.append(round(conf,2))         

        return measures



    def computeSimilarity(self, allAntecedents, antecedentsArray, nbantecedent):
        onlySameSize = [x for x in allAntecedents if len(x) >= len(antecedentsArray)]
        maxSimilarity = 0
        for antecedentIndex in range(len(onlySameSize)):
            antecedents = onlySameSize[antecedentIndex]
            similarity = 0
            for item in antecedents:
                if item in antecedentsArray:
                    similarity += 1
            similarity /= nbantecedent
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        return maxSimilarity

    def generateRules(self, data, numberOfRules=2, nbAntecedent=2, path='',minSupp=0,minConf=0):
        print('begin rules generation')
        timeCreatingRule = 0
        timeComputingMeasure = 0
        for consequent in tqdm.tqdm(range(self.dataSize)):
            if consequent % 10 == 0:
                print('progress : ' + str(round(consequent / self.dataSize, 2)*100) + ' %')
            allAntecedents = []
            for j in range(numberOfRules):
                antecedentsArray = []
                for i in range(nbAntecedent):
                    t1 = time.time()
                    consequentArray = np.zeros(self.dataSize)
                    consequentArray[consequent] = 1
                    consequentArray[antecedentsArray] = 1
                    consequentArray = torch.tensor(consequentArray).to(self.device)
                    consequentArray = consequentArray.unsqueeze(0)
                    output = self.model(consequentArray.float())
                    output = output.cpu()
                    output = np.array(output.detach().numpy())
                    output = pd.DataFrame(output.reshape(self.dataSize, -1))
                    potentialAntecedentsArray = output[0].nlargest(len(data.loc[0]))
                    for antecedent in potentialAntecedentsArray.keys():
                        potentialAntecedents = copy.deepcopy(antecedentsArray)
                        potentialAntecedents.append(antecedent)
                        potentialAntecedents = sorted(potentialAntecedents)
                        if antecedent != consequent and antecedent not in antecedentsArray and self.computeSimilarity(
                            allAntecedents, potentialAntecedents, nbAntecedent) <= self.likeness:
                            antecedentsArray.append(antecedent)
                            break
                    t2 = time.time()
                    measures = self.computeMeasures(copy.deepcopy(antecedentsArray),consequent,data)
                    t3 =time.time()
                    ruleProperties = [list(sorted(copy.deepcopy(antecedentsArray))),[consequent]]
                    ruleProperties += measures
                    self.results.append(ruleProperties)
                    allAntecedents.append(sorted(copy.deepcopy(antecedentsArray)))
                    timeCreatingRule+=t2-t1
                    timeComputingMeasure+=t3-t2
        df = pd.DataFrame(self.results,columns=['antecedent','consequent','support','confidence'])
        # print("shape of df before removing duplicates",df.shape)
        # df = df[~df.duplicated( keep="first")]


        # print("shape of df after removing duplicates",df.shape)
        new_list = []
        for row,value in df.iterrows():
            if value["support"] >= minSupp and value["confidence"]>=minConf:
                new_list.append(value)
        df = pd.DataFrame(new_list)
        print("df last value is ",df.iloc[-1]["consequent"])
        print("df last value is ",type(df.iloc[-1]["consequent"]))

        print("shape of df before readmission",df.shape)
        df = df[df['consequent'].apply(lambda x: len(x) == 1 and (x[0] == self.dataSize-1))]

        # df =  df[(df['consequent'] == [147]) | (df['consequent'] == [148])]
        print("shape of df after readmission",df.shape)

        # Create a dictionary mapping indices to column names
        index_to_column = {i: data.columns[i] for i in range(len(data.columns))}

        # Function to replace indices with column names
        def replace_indices_with_names(indices_list):
            return [index_to_column[idx] for idx in indices_list]

        # Apply the function to the antecedent column

        # df['consequent'] = df['consequent'].apply(literal_eval)
        # df['antecedent'] = df['antecedent'].apply(literal_eval)

        df['antecedent'] = df['antecedent'].apply(replace_indices_with_names)
        df['consequent'] = df['consequent'].apply(replace_indices_with_names)
        df["antecedent"] = df["antecedent"].apply(tuple)
        df["consequent"] = df["consequent"].apply(tuple)
        df.drop_duplicates(inplace=True, subset=["antecedent", "consequent"])
        # df.drop_duplicates(inplace=True,subset=["antecedent","consequent"])
        df = df.reset_index(drop=True)
        df.to_csv(path,index=False)
        return timeCreatingRule, timeComputingMeasure






