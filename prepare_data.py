import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

def scale(df, ordinal_features, target):
    # store target into separate dataframe before scaling
    df_target = df[target]
    # remove target feature
    df.drop(columns=target, inplace=True)
    # initialize ordinal_variables
    ordinal_variables = []
    # count number of different unique values for each feature
    df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
    # retrieve binary variables/features
    binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
    # retrieve categorical variables/features
    categorical_variables = list(df_uniques[(df_uniques['Unique Values'] <= 10) & (df_uniques['Unique Values'] > 2)].index)
    # retrieve ordinal variables/features
    if ordinal_features is not None:
        ordinal_variables.append(ordinal_features)
    # retrieve numeric variables/features
    numeric_variables = list(set(df.columns) - set(ordinal_variables) - set(categorical_variables) - set(binary_variables))
    # Encode ordinal features
    Oe = OrdinalEncoder()
    df[ordinal_variables] = Oe.fit_transform(df[ordinal_variables])
    # Encode binary features
    lb = LabelBinarizer()
    for column in binary_variables:
        df[column] = lb.fit_transform(df[column])
    # Encode numeric features
    mm = MinMaxScaler()
    for column in [ordinal_variables + numeric_variables]:
        df[column] = mm.fit_transform(df[column])
    # Encode categorical features
    df = pd.get_dummies(df, columns = categorical_variables, drop_first=True)

    return df.join(df_target) # return modified with previously saved target serie

