import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

current_dir = os.path.dirname(os.path.abspath(__file__))

data_clean = pd.read_csv(os.path.join(current_dir, 'Dataset', 'data_clean.csv'))

data_clean = pd.get_dummies(data_clean, columns=['early_access'], drop_first=True)

mlb = MultiLabelBinarizer()

genres_encoded = pd.DataFrame(mlb.fit_transform(data_clean['genres']), columns=mlb.classes_)
data_clean = pd.concat([data_clean, genres_encoded], axis=1)

data_clean = data_clean.drop(columns=['genres'])


def predic(release_year: int, earlyaccess: bool, genres: list):

    X = data_clean.drop(['price'], axis=1)
    y = data_clean['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest = RandomForestRegressor(random_state=42)

    random_forest.fit(X_train, y_train)

    input_data = pd.DataFrame({'release_year': [release_year],
        'early_access_True': [1 if earlyaccess else 0]})

    for genero in genres:
        if genero in mlb.classes_:
            input_data[genero] = 1

    y_pred = random_forest.predict(input_data)
    y_test_pred = random_forest.predict(X_test)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    return {'Precio': y_pred[0], 'RMSE':rmse}