import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

train_data = pd.read_csv("train.csv", parse_dates=["pickup_datetime"])
test_data = pd.read_csv("test.csv", parse_dates=["pickup_datetime"])

print("train:", train_data.shape)
print("test: ", test_data.shape)

