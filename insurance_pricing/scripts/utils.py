import pandas as pd


# One hot encoding for categorical variables
def OH_df(X):
    one_hot = pd.get_dummies(X["Marital_Status"])
    X = X.drop("Marital_Status", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["State"])
    X = X.drop("State", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Coverage_Type"])
    X = X.drop("Coverage_Type", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Education"])
    X = X.drop("Education", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Employment_Status"])
    X = X.drop("Employment_Status", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Location_Code"])
    X = X.drop("Location_Code", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Sales_Channel"])
    X = X.drop("Sales_Channel", axis=1)
    X = X.join(one_hot)

    one_hot = pd.get_dummies(X["Policy_Type"])
    X = X.drop("Policy_Type", axis=1)
    X = X.join(one_hot)
    return X
    