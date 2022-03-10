from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
            ])
        self.pipeline=pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data()

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # build pipeline
    trainer=Trainer(X_train,y_train)
    trainer.set_pipeline()

    # train the pipeline
    trainer.run()

    # evaluate the pipeline
    trainer.evaluate(X_test, y_test)
