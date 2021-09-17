import utils
import config
import model
import engine
import loss
import metric

import argparse
import torch
import torch.optim as optim
import os
import pickle

def run():
    train_dataset, test_dataset, total_nos_labels, x_dimension, mid_embedding_size, resultant_embedding_size = utils.load_dataset(config.DATASET)

    # Making Dataloader
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    testDataLoader  = torch.utils.data.DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1)

    device = torch.device(config.DEVICE)

    # Defining Model
    print("Making Model!!...")
    dxmlModel = model.DXML(x_dimension, mid_embedding_size, resultant_embedding_size)
    dxmlModel.to(device)

    # Declaring the optimizer and its parameters
    optimizer_param = list(dxmlModel.parameters())
    num_train_steps = int( len(trainDataLoader) * config.EPOCHS)
    optimizer       = optim.SGD(dxmlModel.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)


    if config.doTrain:
        print("Going into training")
        for epoch in range(config.EPOCHS):
            a = list(dxmlModel.parameters())[0].clone()
            training_loss = engine.train(trainDataLoader, dxmlModel, optimizer, device)
            print("Epoch: ",epoch+1," Loss: ",training_loss)
            b = list(dxmlModel.parameters())[0].clone()
            print('Are param equal? ',torch.equal(a.data, b.data))


        print("\nTraining is finished!!")
        os.makedirs(os.path.dirname(config.MODEL_SAVED.format(config.DATASET)), exist_ok=True)
        torch.save(dxmlModel.state_dict(), config.MODEL_SAVED.format(config.DATASET))
        print('Model is saved at: ',config.MODEL_SAVED.format(config.DATASET))

    
    
    # Generating FX - set of all fx and clustering them
    print('\nLoading the model...')
    dxmlModel.load_state_dict(torch.load(config.MODEL_SAVED.format(config.DATASET)))
    engine.generate_cluster_FX(trainDataLoader, dxmlModel, total_nos_labels)


    # Evaluating the model
    print('\nEvaluation started')
    final_targets, final_outputs = engine.eval(testDataLoader, dxmlModel, device)

    print('Saving the results')
    os.makedirs(os.path.dirname(config.PREDICTIONS_PATH.format(config.DATASET)), exist_ok=True)
    with open(config.PREDICTIONS_PATH.format(config.DATASET), 'wb') as f:
        pickle.dump(final_targets, f) # First dump the ground truth 
        pickle.dump(final_outputs, f) # Now dump the predicted output

    print('\nNow starting Evaluation...')
    metric.start_prediction_eval()







def small_configurations():
    config.MID_EMBEDDINGS_SIZE = 256
    config.END_EMBEDDINGS_SIZE = 100


def big_configurations():
    config.MID_EMBEDDINGS_SIZE = 512
    config.END_EMBEDDINGS_SIZE = 300


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest="dataset", required=True, type=str, help="Enter dataset name")
    parser.add_argument('--doTrain', dest="doTrain", required=True, type=str, help="Enter True or False")
    args   = parser.parse_args()

    if args.dataset.lower() == "Bibtex".lower():
        config.DATASET = 'Bibtex'
        small_configurations()
    elif args.dataset.lower() == "Delicious".lower():
        config.DATASET = 'Delicious'
        small_configurations()
    elif args.dataset.lower() == "MediaMill".lower():
        config.DATASET = 'Mediamill'
        small_configurations()
    else:
        print(args.dataset+' is not a valid dataset name. Ending..')
        exit()

    if args.doTrain.lower() == "True".lower():
        config.doTrain = True
    else:
        config.doTrain = False
    run()