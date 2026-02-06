import torch.nn as nn

class FashionMLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, num_classes=10):
        super(FashionMLP, self).__init__()

        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.output = nn.Linear(hidden2_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        logits = self.output(x)
        return logits