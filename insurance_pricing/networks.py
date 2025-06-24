import torch.nn as nn

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, shape):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(shape, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        return self.sigmoid(self.linear(X))


# Define the pricing model
class PricingModel(nn.Module):
    def __init__(self, shape):
        super(PricingModel, self).__init__()
        self.linear = nn.Linear(shape, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, X):
        return self.linear2(self.relu(self.linear(X)))