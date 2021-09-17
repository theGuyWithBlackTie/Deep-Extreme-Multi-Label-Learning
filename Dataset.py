from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torch
import pandas as pd
import numpy as np

import config

class XMTCDataset(Dataset):
    def __init__(self, dataset_file, total_data_points, x_dimension, y_dimension, isTrainDataset, embedding_file=None):
        self.Xsamples = np.zeros((total_data_points, x_dimension), dtype=np.int8)
        self.Ysamples = np.zeros((total_data_points, y_dimension), dtype=np.int8)
        self.YEmbed   = np.zeros((total_data_points, y_dimension), dtype=np.int8)

        print('\nDataset\'s shapes are: X->',self.Xsamples.shape,' Y->',self.Ysamples.shape,' Embeddings->',self.YEmbed.shape)

        embedding_np  = None
        if isTrainDataset:
            if embedding_file is None:
                print("Embedding file is not provided")
                exit()
            else:
                embedding_df = pd.read_csv(embedding_file, sep=' ', header=None)
                embedding_df = embedding_df.T
                embedding_np = embedding_df.to_numpy()

        # Reading file and loading the dataset into 'Xsamples'
        with open(dataset_file, encoding='utf-8') as file:
            line = file.readline()
            row  = 0
            while line:
                line     = line.split(' ')
                features = line[1:]
                labels   = line[0].split(',')

                try:
                    for eachLabel in labels:
                        self.Ysamples[row][int(eachLabel)] = 1

                    for eachFeature in features:
                        eachFeature = eachFeature.split(':')
                        try:
                            self.Xsamples[row][int(eachFeature[0])] = float(eachFeature[1])
                        except:
                            print('Error: row->',row,' eachFeature[0]',eachFeature[0], ' labels ',labels)
                            exit()
                except:
                    print('Row {} has problem with label index'.format(row+1))
                    
                line = file.readline()
                row  = row + 1

        if isTrainDataset:
            print('Projecting original label matrix on embedding matrix..')
            each_row_sum  = self.Ysamples.sum(axis=1) # Taking the sum in each label vector to get nos. of non zero elements
            each_row_sum  = each_row_sum.reshape(total_data_points,1)
            self.YEmbed   = np.dot(embedding_np, self.Ysamples.T)
            self.YEmbed   = self.YEmbed.T
            self.YEmbed   = np.divide(self.YEmbed, each_row_sum)        
            print('Projection is done. Re-checking shapes now. X->',self.Xsamples.shape,' Y->',self.Ysamples.shape,' Embeddings->',self.YEmbed.shape)
                

    
    def __len__(self):
        return len(self.Xsamples)

    def __getitem__(self, idx):
        return {
            config.X: torch.tensor(self.Xsamples[idx]),
            config.TARGET: torch.tensor(self.Ysamples[idx]),
            config.EMBED: torch.tensor(self.YEmbed[idx])
        }
