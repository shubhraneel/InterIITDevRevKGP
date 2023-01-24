import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = np.load("cosine_similarities.npz", allow_pickle=True)

cos_sims = []
for i in data["arr_0"]:
    temp = []
    for x in i:
        temp2 = []
        for y in x:
            temp2.append(1 - y[0])
        temp.append(temp2)
    cos_sims.append((temp))
cos_sims = np.array(cos_sims)

max_len = 27
for i in cos_sims:
    if len(i) != 27:
        for num in range(27 - len(i)):
            i.append([1])

for i in cos_sims:
    i = np.array(i)

df2 = pd.DataFrame()
df2["cosine_sims"] = pd.DataFrame(cos_sims)

for i in range(27):
    for k in range(len(df2)):
        df2.loc[k, "column_cos_" + "%s" % i] = cos_sims[k][i][0]

dataframe = pd.read_csv("train_data_sentidx.csv")

df2["Sentence Index"] = dataframe["Sentence Index"]
df2 = df2.dropna()
df2 = df2.drop(["cosine_sims"], axis=1)

labels2 = df2["Sentence Index"]
df2 = df2.drop(["Sentence Index"], axis=1)

train_x, test_x, train_y, test_y = train_test_split(
    df2, labels2, train_size=0.8, random_state=42
)

mul_lr = linear_model.LogisticRegression(multi_class="multinomial", solver="newton-cg")
mul_lr.fit(train_x, train_y)

print(
    "Multinomial Logistic regression Train Accuracy : ",
    metrics.f1_score(train_y, mul_lr.predict(train_x), average="weighted"),
)
print(
    "Multinomial Logistic regression Test Accuracy : ",
    metrics.f1_score(test_y, mul_lr.predict(test_x), average="weighted"),
)

rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
rf.fit(train_x, train_y)

print(
    "RF Train Accuracy : ",
    metrics.f1_score(train_y, rf.predict(train_x), average="weighted"),
)
print(
    "RF Test Accuracy : ",
    metrics.f1_score(test_y, rf.predict(test_x), average="weighted"),
)

model = xgb.XGBClassifier()
param_dist = {
    "max_depth": [3, 5, 10],
    "min_child_weight": [1, 5, 10],
    "learning_rate": [0.07, 0.1, 0.2],
}

# run randomized search
grid_search = GridSearchCV(model, param_grid=param_dist, cv=3, verbose=5, n_jobs=-1)
grid_search.fit(train_x, train_y)

xg = xgb.XGBClassifier(max_depth=5)
xg.fit(train_x, train_y)

print(
    "XGB Train Accuracy : ",
    metrics.f1_score(train_y, xg.predict(train_x), average="weighted"),
)
print(
    "XGB Test Accuracy : ",
    metrics.f1_score(test_y, xg.predict(test_x), average="weighted"),
)
