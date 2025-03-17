import numpy as np
import pandas as pd
import lightgbm as lgb

alpha_init = np.array([4.75, 0.5, 0.25, 4.75, 0.5, 0.25])

beta_init = np.array([14.25, 1.5, 0.75, 14.25, 1.5, 0.75])

print(alpha_init)
print(alpha_init / (alpha_init + beta_init))
example = np.array([1, 2, 3, 1, 2, 3])
print(example)
# Change mult to !=1 to have this variable have an effect in alpha
mult = 1
x = np.array([1.0, 1.0, 1.0, mult, mult, mult])

# Carry x's effect to modify alpha
alpha_init = alpha_init * x
print(alpha_init)
del mult

# Oversampling factor
k = 5000

# Build dataframe
df = pd.DataFrame(
    {
        "alpha": np.array(list(alpha_init) * k),
        "beta": np.array(list(beta_init) * k),
        "x": [e for e in np.array(list(x) * k)],
        "example": [int(e) for e in np.array(list(example) * k)],
    }
)
print(df.head(10))

df["theta"] = np.random.beta(df["alpha"], df["beta"])
print(df.head())

# Define if we should be using random censoring
random_censoring = True

# Define a number of maximum months a user can be observed for (e.g. if maxT==12 it would mean our oldest customers have been with us for 1 year)
maxT = 12

# Build the number of observed months
if random_censoring:
    df["obs_months"] = np.random.randint(1, maxT + 1, df.shape[0])
else:
    df["obs_months"] = maxT

print(df.head())

# Initialize user tenure to 0
df["tenure"] = 0


# Update tenure function
def tenure_func(c, theta, j, tenure):
    """For every month, sample if the user churns, and return the month in which they churn"""

    out = 0

    # If the user had not churn before and they churn now, we return the month at which they churn
    if tenure == 0 and c <= theta:
        out = j

    # If the user had already churn (tenure!=0) we return whatever month they had already churn at.
    # If the user has not churn yet but the churn wasn't drawn (c>theta), we return their tenure=0
    else:
        out = tenure

    return out


# For each month, draw whether the user churns and update their tenure
for j in range(1, maxT + 1):
    df["c_tmp"] = np.random.random_sample(df.shape[0])
    df["tenure"] = df.apply(
        lambda row: tenure_func(row["c_tmp"], row["theta"], j, row["tenure"]), axis=1
    )


df.drop(columns="c_tmp", inplace=True)
print(df.head())
print(df.tenure.value_counts())

# Define the churn event
df["observed"] = df.apply(
    lambda row: 1 if (row["tenure"] > 0 and row["tenure"] <= row["obs_months"]) else 0,
    axis=1,
)

# Modify tenures of 0 to maximum tenure (our user has been with us maxT periods and counting)
df["tenure"] = df["tenure"].apply(lambda x: maxT if x == 0 else x)
print(df.head(10))

df["tenure"] = df.apply(lambda row: np.min([row.tenure, row.obs_months]), axis=1)

num_months_to_study = 3

# Total number of months of observation is upper bounded by our number of months to study
df["future_months"] = df["tenure"].apply(
    lambda x: num_months_to_study if x > num_months_to_study else x
)


# My implementation, censored could also be if we simply have not observed the user for at least num_months_to_study
df["is_censored"] = df.apply(
    lambda row: (
        1
        if (
            row["tenure"] > num_months_to_study
            or (row["tenure"] == row["obs_months"])
            and not row["observed"]
        )
        else 0
    ),
    axis=1,
)
"""
# Original implementation, which seems to not be accurate when we have censoring? Something I'm missing here
df['is_censored']   = df['tenure'].apply(lambda x: 1 if x > num_months_to_study else 0)
"""
print(df.head())

df["event_col"] = 1 - df["is_censored"]

# What are these 2 for?!
df["weight"] = 1
df["lr_label"] = (df["future_months"] > 1).astype(int)
print(df.head())

msk = np.random.rand(len(df)) < 0.6
df_train = df[msk]
df_test = df[~msk]
# check probs
print(df.groupby(by="example")["lr_label"].mean())

features2include = ["example", "x"]
categorical_feature = [0]


dtrain_bl = lgb.Dataset(
    data=df_train[features2include],
    label=df_train["future_months"],
    weight=df_train["is_censored"],
    categorical_feature=categorical_feature,
)

n_trees = 100

common_params = {
    "bagging_fraction": 1,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 876,
    "max_depth": -1,
    "learning_rate": 0.05,
    "num_threads": 12,
    "num_leaves": 40,
    "min_data_in_leaf": 1,
    "lambda_l1": 0,
    "lambda_l2": 1,
    "min_sum_hessian_in_leaf": 1e-3,
}

params_bl = {"objective": "sbg", "metric": "sbg", "num_class": 2}

params = {**params_bl, **common_params}

print("Training model")
model_bl = lgb.train(
    params=params,
    train_set=dtrain_bl,
    num_boost_round=n_trees,
    valid_sets=dtrain_bl,  # eval training data
)


def bl_predict_survival(model, data, num_trees, max_horizon):

    # Run model predictions for the dataset for number of rounds selected (by default, the training rounds)
    ab = model.predict(data, num_iteration=num_trees)

    # Unpack model predictions, which are an alpha and a beta for each observation, with a total of N predictions (size of inference dataset)
    alpha, beta = ab[:, 0], ab[:, 1]
    N = len(alpha)

    # Initialize predicted survival sequences for each sample, where p in t0 is the mean of the predicted Beta distribution at the customer level
    curves = np.zeros((N, max_horizon + 1), dtype=np.float32)
    p = alpha / (alpha + beta)
    s = 1 - p
    curves[:, 0] = 1
    curves[:, 1] = s.astype(dtype=np.float32)

    # For each time horizon, calculate the subsequent vectors p and s (NO CENSORING?)
    for i in range(2, max_horizon + 1):
        p = p * (beta + i - 2) / (alpha + beta + i - 1)
        s = s - p
        curves[:, i] = s.astype(dtype=np.float32)

    return curves, alpha, beta


print("Predicting survival")

curves, alpha, beta = bl_predict_survival(
    model_bl, df_test[features2include], n_trees, maxT
)

print(curves, alpha, beta)
