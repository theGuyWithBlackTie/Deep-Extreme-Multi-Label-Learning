import random
import linecache
import os
import subprocess

import config
import Dataset


'''
This function generates the training and testing data file.
Internally calls function to generate the line nos for splitting.
'''
def generate_train_test_files(dataset_name,train_file, test_file):
    train_split_file = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_trSplit.txt'
    test_split_file  = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_tstSplit.txt'

    train_indexes = []
    test_indexes  = []
    split_index   = random.randint(0,9)

    train_indexes = small_dataset_split_list_generation(train_split_file, split_index)
    test_indexes  = small_dataset_split_list_generation(test_split_file, split_index)

    print("Got the line numbers for split.\nNow generating "+dataset_name+"_train.txt and "+dataset_name+"_test.txt files")
    dataset_file       = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+"_data.txt"
    train_file_handler = open(train_file, 'w+')
    test_file_handler  = open(test_file, 'w+')
    for each_line in train_indexes:
        line     = linecache.getline(dataset_file, each_line)
        train_file_handler.write(line)
    for each_line in test_indexes:
        line     = linecache.getline(dataset_file, each_line)
        test_file_handler.write(line)
    train_file_handler.close()
    test_file_handler.close()
    print("Generated "+dataset_name+"_train.txt and "+dataset_name+"_test.txt files")




'''
This function is used only for small dataset.
It generates the line indexes to use for splitting the mail file of small dataset
'''
def small_dataset_split_list_generation(file, split_index):
    index_list = []
    with open(file, encoding='utf-8') as f:
        line = f.readline()
        while line:
            index_list.append(int(line.split(' ')[split_index])+1)
            line = f.readline()
    return index_list



'''
Main function in this file to load the dataset.
Call will come here first
'''
def load_dataset(dataset_name = 'Bibtex'):
    bow_dimension     = 0
    total_nos_labels  = 0
    total_data_points = 0
    end_embedding_size= 0
    mid_embedding_size=0
    
    if dataset_name == 'Bibtex' or dataset_name == 'Delicious' or dataset_name == 'Mediamill':
        print('Small Dataset')
        train_file = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_train.txt'
        test_file  = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_test.txt'

        if not os.path.isfile(train_file) or not os.path.isfile(test_file):
            print("Training & Testing File doesn't exists")
            print("Generating "+dataset_name+"_train.txt and "+dataset_name+"_test.txt files")
            generate_train_test_files(dataset_name, train_file, test_file)
        
        x_datapoints_len, y_datapoints_len = get_total_data_points(dataset_name)
        with open(config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_data.txt') as file:
            line = file.readline()  # Read only first line

            # Loading the numbers related to the dataset
            line              = line.split(' ')
            total_data_points = int(line[0])
            bow_dimension     = int(line[1])
            total_nos_labels  = int(line[2])

        end_embedding_size = config.END_EMBEDDINGS_SIZE
        mid_embedding_size = config.MID_EMBEDDINGS_SIZE

        # Checking whether embedding exists
        embedding_file = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'.embeddings'
        if not os.path.isfile(embedding_file):
            print('Embeddings of selected dataset '+dataset_name+' doesn\'t exists. Generating now')
            generate_embeddings(dataset_name, total_nos_labels,embedding_file,end_embedding_size)

        
        train_dataset = Dataset.XMTCDataset(train_file,x_datapoints_len, bow_dimension, total_nos_labels, True, embedding_file=embedding_file)
        
        # test_dataset  = Dataset.XMTCDataset(test_file,y_datapoints_len, bow_dimension, total_nos_labels, False) Use this when you don't want to have test set's loss
        test_dataset  = Dataset.XMTCDataset(test_file,y_datapoints_len, bow_dimension, total_nos_labels, True, embedding_file=embedding_file)

    return train_dataset, test_dataset,total_nos_labels, bow_dimension, mid_embedding_size, end_embedding_size




'''
Function which generates the embeddings of the label graph.
It automatically calls deepwalk subprocess.
Make sure that deepwalk is installed in the system.
'''
def generate_embeddings(dataset_name, total_nos_labels,embedding_file,embedding_size):

    # Creating label adjacency file first
    adjacency_file = config.DATAFOLDER+dataset_name+"-small/"+dataset_name+".txt" # For some reason Python doesn't create file with custom extensions hence changed .adjlist to .txt
    write_to_file  = open(adjacency_file, 'w+')
    label_list     = []
    with open(config.DATAFOLDER+dataset_name+"-small/"+dataset_name+"_train.txt", encoding="utf-8") as f:
        line = f.readline()
        while line:
            labels = line.split(' ')[0].split(',')
            label_list.extend(labels)
            for i in range(0, len(labels)):
                for j in range(0, len(labels)):
                    if i != j:
                        line_to_write = "{} {}".format(labels[i], labels[j])
                        line_to_write = line_to_write.strip()+'\n'
                        write_to_file.write(line_to_write)
            line = f.readline()

    print('Generating the embeddings')
    command = "deepwalk --format edgelist --input {} --output {} --representation-size 100  --max-memory-data-size 0 --walk-length {} --window-size {} --workers 1 --number-walks {}".format(adjacency_file,embedding_file+"temp", config.WALK_LENGTH, config.WINDOW_SIZE, config.NUMBER_WALK)
    result  = subprocess.check_output(command, shell=True)
    print(result.decode("utf-8"))

    # Arranging the embeddings in ascending order of Labels and traversing it
    embed_file = open(embedding_file, "w")
    embed_dict = {}
    with open(embedding_file+"temp", encoding="utf-8") as f:
        line = f.readline()
        line = f.readline()
        while line:
            line  = line.split(' ')
            index = line[0]
            embed = line[1:]
            embed_dict[int(index)] = embed
            line = f.readline()
    
    for index in range(0,total_nos_labels):
        line = embed_dict.get(index)
        if line is None:
            embed_file.write(' '.join('0'*embedding_size)+'\n')
        else:
            embed_file.write(' '.join(line))
    embed_file.close()
    
    os.remove(embedding_file+"temp")
    print('Embeddings are generated')




'''
A utility function to return totoal nos. of datapoints from train and test dataset.
'''
def get_total_data_points(dataset_name):
    train_split_file = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_trSplit.txt'
    test_split_file  = config.DATAFOLDER+dataset_name+'-small/'+dataset_name+'_tstSplit.txt'

    x_datapoints_len = 0
    with open(train_split_file, encoding="utf-8") as f:
        line = f.readline()
        while line:
            x_datapoints_len = x_datapoints_len + 1
            line = f.readline()

    y_datapoints_len = 0
    with open(test_split_file, encoding="utf-8") as f:
        line = f.readline()
        while line:
            y_datapoints_len = y_datapoints_len + 1
            line = f.readline()
    return x_datapoints_len, y_datapoints_len