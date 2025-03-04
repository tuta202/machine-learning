import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_ts_data(data, window_size=5):
  i = 1
  while i < window_size:
    data["co2_{}".format(i)] = data["co2"].shift(-i)
    i+=1
  data["target"] = data["co2"].shift(-i)
  data = data.dropna(axis=0)
  return data

data = pd.read_csv("co2.csv")

data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()

data = create_ts_data(data)

# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# plt.show()

x = data.drop(["time", "target"], axis=1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
x_test = x[int(num_samples*train_ratio):]
y_test = y[int(num_samples*train_ratio):]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print(y_predict)

# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))

# LinearRegression
# MAE: 0.3605603788359208
# MSE: 0.22044947360346367
# R2: 0.9907505918201437

# RandomForestRegressor
# MAE: 5.601491228070157
# MSE: 49.30942553070146
# R2: -1.0688777178396438

# fig, ax = plt.subplots()
# ax.plot(data["time"][:int(num_samples * train_ratio)], data["co2"][:int(num_samples * train_ratio)], label="train")
# ax.plot(data["time"][int(num_samples * train_ratio):], data["co2"][int(num_samples * train_ratio):], label="test")
# ax.plot(data["time"][int(num_samples * train_ratio):], y_predict, label="prediction")
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()

# current_data = [380.5, 390, 390.2, 390.4, 393]
# for i in range(10):
#   prediction = reg.predict([current_data])[0]
#   print("CO2 in week {} is {}".format(i+1, prediction))
#   current_data = current_data[1:] + [prediction]
