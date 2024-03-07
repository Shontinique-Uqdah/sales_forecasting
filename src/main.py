# Python version
import sys
print('Python: {}'.format(sys.version))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))

from os.path import exists
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType
from catboost import CatBoostRegressor
from catboost.utils import eval_metric

df_path = "train.csv.zip" if exists("train.csv.zip") else "https://github.com/upgini/upgini/raw/main/notebooks/train.csv.zip"
df = pd.read_csv(df_path)
df = df.sample(n=10_000, random_state=0)
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)

df["date"] = pd.to_datetime(df["date"])

df.sort_values("date", inplace=True)
df.reset_index(inplace=True, drop=True)
df.head()

#training set includes anything before Jan 2017; test set anything after

train = df[df["date"] < "2017-01-01"]
test = df[df["date"] >= "2-17-01-01"]

#all other labels (date, store-id, item-id) are features
train_features = train.drop(columns=["sales"])
#sales is train target
train_target = train["sales"]
test_features = test.drop(columns=["sales"])
test_target = test["sales"]

#enrich data with upgini

enricher = FeaturesEnricher(
    search_keys = {
        "date": SearchKey.DATE,
        },
    cv = CVType.time_series
    )
enricher.fit(train_features,
             train_target,
             eval_set=[(test_features,test_target)])


#define catboost model
model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0)

#calculate metrics before and after enrichment; compare baseline vs enriched mean absolute percentage error
enricher.calculate_metrics(
    train_features, train_target, 
    eval_set=[(test_features,test_target)],
    estimator = model,
    scoring = "mean_absolute_percentage_error"
    )

#feed in existing training dataset and get enriched trained features dataset; input = True maintains our initial columns
enriched_train_features = enricher.transform(train_features, keep_input=True)
enriched_test_features = enricher.transform(test_features,keep_input=True)
enriched_train_features.head()

#train model on train features and train target.
model.fit(train_features, train_target)
#Predict with test features
predictions = model.predict(test_features)
#evaluate with test target values, predictions, and "SMAPE"; error rate with original dataset
eval_metric(test_target.values, predictions, "SMAPE")


#train model on enriched train features and train target.
model.fit(enriched_train_features, train_target)
#Predict with enriched test features
enriched_predictions = model.predict(enriched_test_features)
#evaluate with test target values, predictions, and "SMAPE"; error rate with original dataset
eval_metric(test_target.values, enriched_predictions, "SMAPE")