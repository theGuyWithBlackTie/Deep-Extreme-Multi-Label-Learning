# Contains all static and live configurations about the project
import torch

# paths
DATASET    = 'Bibtex'  # Default dataset is set to Bibtex dataset
DATAFOLDER = 'data/'
MODEL_SAVED= 'outputs/{}/model/state_dict_model.pt'
PREDICTIONS_PATH = 'outputs/{}/predictions/final_results_and_targets.pkl'
METRICS_PATH = 'outputs/{}/metric/metric.txt'
DATA_EMBEDDINGS = 'outputs/{}/data_embeddings/data.embeddings'
KMEANS_MODEL    = 'outputs/{}/model/kmeans_prediction.pkl'

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE  = 16

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MID_EMBEDDINGS_SIZE = 256  # Default value. Big Dataset - 512
END_EMBEDDINGS_SIZE = 100  # Default value. Big Dataset - 300

EPOCHS        = 70 #70
LEARNING_RATE = 0.03  # 0.015 # 1e-7 
MOMENTUM      = 0.9 # 0.9
WEIGHT_DECAY  = 1e-4 #1e-4 #0.0005

TOP_P         = 10
TOP_K         = 7 # 20 works best 5 was before
CLUSTERS_NUM  = 50

doTrain       = True


X      = "X"
TARGET = "Y"
EMBED  = "EMBED"

WALK_LENGTH = 30
WINDOW_SIZE = 2
NUMBER_WALK = 80
