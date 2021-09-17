from tqdm import tqdm
import torch
from sklearn.cluster import KMeans
import numpy as np
import pickle
import numpy as np
import math
import torch.nn.functional as F

import loss
import config


def loss_fn(output, target):
    return F.smooth_l1_loss(output, target)

def train(dataloader, model, optimizer, device):
    model.train()
    result_loss = 0

    for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        input  = d[config.X]
        target = d[config.EMBED]

        input  = input.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(input)

        loss   = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        result_loss += loss

    return result_loss/len(dataloader)




# Function to generate the set of embeddings of training dataset - FX
def generate_cluster_FX(dataloader, model, total_nos_labels):
    model.eval()
    FX      = np.zeros((1,config.END_EMBEDDINGS_SIZE))
    targets = np.zeros((1, total_nos_labels))
    print('\nGenerating FX...')
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            input  = d[config.X]
            target = d[config.TARGET]

            input  = input.to(config.DEVICE, dtype=torch.float)
            output = model(input)

            FX     = np.append(FX, output.cpu().detach().numpy(), 0)
            targets= np.append(targets, target.cpu().detach().numpy(),0)

    FX      = FX[1:,:]   # Removing first dummy row of zeros
    targets = targets[1:,:]

    # Saving the FX array (Required only if plotting FX)
    #a_file = open('FX.txt', 'w')
    #for row in FX:
    #    np.savetxt(a_file, row.reshape(1, row.shape[0]))
    #a_file.close()
    # End here

    print('FX.shape: ',FX.shape)
    kMeans_model =  generate_clusters(FX)

    # Saving the KMeans Model
    with open(config.KMEANS_MODEL.format(config.DATASET), 'wb') as pickle_file:
        pickle.dump(kMeans_model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(FX, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(targets, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nKMeans Model, FX and targets are saved at ',config.KMEANS_MODEL.format(config.DATASET))




# Function to generate the clusters from the FX
def generate_clusters(FX):
    print('\nStarting clustering...')
    kMeans_model = KMeans(n_clusters = config.CLUSTERS_NUM, max_iter=1000, random_state=42)
    kMeans_model.fit(FX)
    return kMeans_model




# Function to evaluate the model
def eval(dataloader, model, device):
    model.eval()
    print('Loading the KMeans prediction model...')
    final_outputs = []
    final_targets = []
    loss         = 0 # debugging
    with open(config.KMEANS_MODEL.format(config.DATASET), 'rb') as pickle_file:
        kMeans_model = pickle.load(pickle_file)
        FX           = pickle.load(pickle_file)
        targets      = pickle.load(pickle_file)
        
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            test_input  = d[config.X]
            test_target = d[config.TARGET]
            test_embed  = d[config.EMBED] # debugging - Added line to get test loss

            test_input  = test_input.to(device, dtype=torch.float)
            test_target = test_target.to(device, dtype=torch.float)
            test_embed  = test_embed.to(device, dtype=torch.float) # debugging - Added line to get test loss

            test_output = model(test_input)
            loss        += loss_fn(test_output, test_embed)
            
            test_output = test_output.cpu().detach().numpy() # Receiving the output from the model and converting it into numpy array
            


            # Finding the nearest cluster
            for eachRow in test_output:
                cluster_no      = find_ZI_star(eachRow, kMeans_model)
                cluster_indices = ClusterIndicesNumpy(cluster_no, kMeans_model.labels_)
                k_nn_elems      = find_k_nn(cluster_indices, eachRow, FX)
                y_cap           = generate_y_cap(k_nn_elems, targets)
                final_outputs.append(y_cap)

            final_targets.extend(test_target.cpu().detach().numpy().tolist())
    print('Eval Loss: ',loss/len(dataloader))
    return final_targets, final_outputs




# Function to find closes cluster
def find_ZI_star(vector, kMeans_model):
    all_centroids = kMeans_model.cluster_centers_
    distance      = math.inf
    cluster_no    = math.inf
    for eachElem in range(0, len(all_centroids)):
        eachCentroid = all_centroids[eachElem]
        temp_dist    = np.linalg.norm(vector-eachCentroid)
        if temp_dist < distance:
            cluster_no = eachElem
            distance   = temp_dist
    return cluster_no




def ClusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]



def find_k_nn(cluster_indices, vector, FX):
    elem_dist_dict = {}
    for eachElem in cluster_indices:
        each_cluster_elem = FX[eachElem]
        distance          = np.linalg.norm(vector-each_cluster_elem)
        elem_dist_dict[eachElem] = distance
    
    # Sorting the dict based on distance - value
    sorted_elem_dist_dict = {}
    sorted_keys = sorted(elem_dist_dict, key=elem_dist_dict.get, reverse=False)
    for i in sorted_keys:
        sorted_elem_dist_dict[i] = elem_dist_dict[i]
    return sorted_elem_dist_dict



def generate_y_cap(k_nn_elems, targets):
    top_K = config.TOP_K

    top_K_indices = list(k_nn_elems.keys())[0:top_K]
    all_labels    = []
    for eachElem in top_K_indices:
        row = targets[eachElem]
        all_labels.extend(np.where(row == 1)[0].tolist())
    all_labels = sorted(sorted(all_labels, reverse=True), key=all_labels.count, reverse=True)
    temp     = set()
    temp_add = temp.add
    y_cap    = [x for x in all_labels if not (x in temp or temp_add(x))]
    return y_cap