"""Transform the insurance data for policy optimisation.

One hot encode the categorical variables in the dataset.
Scale the dataset using the MinMaxScaler.
"""
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from utils import OH_df

# Load the data
data = pd.read_csv("../data/atoti/data.csv")

# One hot encode the categorical variables
data = OH_df(data)

# Save the price column
price = data["Price"]
# Save the customer id column
customer_id = data["cust_id"]

# Scale the data
scaler = MaxAbsScaler()

data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Add the price column back to the dataset as unscaled
data["UnscaledPrice"] = price
data["cust_id"] = customer_id

# Position the price column at the end
price = data["Price"]
data = data.drop("Price", axis=1)
data["Price"] = price

# Negate the Sale column
data["Sale"] = 1 - data["Sale"]


# Print rows where one of the values is missing
# Remove these rows
print(data[data.isnull().any(axis=1)])
data = data.dropna()


# Save the transformed data
data.to_csv("../data/atoti/scaled_data.csv", index=False)
