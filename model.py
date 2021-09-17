import torch.nn as nn
import torch.nn.functional as F


class DXML(nn.Module):
    def __init__(self, x_dimension, mid_embedding_size, resultant_embedding_size, dropout = 0.1):
        super(DXML, self).__init__()
        self.W1      = nn.Linear(x_dimension, mid_embedding_size)
        self.RELU    = nn.ReLU()
        self.W2      = nn.Linear(mid_embedding_size, resultant_embedding_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        input  = F.normalize(input, dim=1, p=2) # This is L2 normalization.
        output = self.W1(input)
        output = self.RELU(output)
        output = self.W2(output)
        output = self.dropout(output)

        output = F.normalize(output, dim=1, p=2)  # Removing this increases the precision score by +10 unit

        return output