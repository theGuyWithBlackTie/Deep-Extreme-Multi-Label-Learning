# Deep-Extreme-Multi-Label-Learning
This repository is the implementation of paper: Deep Extreme Multi Label Learning (DXML). Paper Link: https://arxiv.org/abs/1704.03718

## Model Architecture
![Model Architecture](https://github.com/theGuyWithBlackTie/Deep-Extreme-Multi-Label-Learning/blob/main/model%20architecture.png)

The input text (i.e. X-Space) is being converted into the embeddings by linear layers. Similarly, from the label graph, each label's embedding is obtained with [DeepWalk](https://github.com/phanein/deepwalk) algorithm. With the help of smooth-L1 loss function, distances between input embeddings and label embeddings are tried to be minimized.

After training, train embeddings are clustered. For each test record, the closest cluster is found. In the closest cluster, nearest neighbours are found whose original labels are assigned to the test records in an empirical fashion.

## To Run:
```
python app.py --dataset <Bibtex/Delicious/MediaMill> --doTrain <True/False>
```

This paper has used 6 different datasets to show the robustness of the methodology. These 6 different datasets are sub-divided into 2 categories: <b>small</b> and <b>large</b>. Each sub-division consists of 3 datasets each. <i>Small</i> dataset has less number of labels to predict and numbers of records are less as well as compared to <i>large</i> datasets. Due to hardware challenges, this code has only used <i>small</i> dataset.

<b>Small Datasets:</b>
- Bibtex
- Delicious
- MediaMill

The major challenge in replicating this paper's results is getting the right hyperparameters. Hyperparameters are not shared in the paper, and due to this, I was only able to obtain the results for Bibtex dataset. Additionally, the Delicious and MediaMill dataset which was used has has many records which has only input word and more than 10 associated labels which makes it really difficult to make algorithm learn correctly.


