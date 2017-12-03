import seaborn as sns
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


def get_titanic():
    df = sns.load_dataset('titanic')
    encoders = {}

    del df['deck']
    df = df.dropna()

    for i in df.columns:
        value = df[i][1]

        if (not isinstance(value, numpy.int64)) and (not isinstance(value, float)):
            encoder = LabelEncoder()
            encoder.fit(df[i])
            df[i + '_encoder'] = encoder.transform(df[i])

            encoders[i] = encoder

    # get Feature and Target
    features = df[['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_encoder',
                   'embarked_encoder', 'class_encoder', 'who_encoder',
                   'adult_male_encoder', 'embark_town_encoder', 'alive_encoder',
                   'alone_encoder']].copy()
    target = df['survived']

    # split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, target, test_size=0.2)

    return x_train, x_test, y_train, y_test
