import pandas as pd
from src.features import split_features_target

def test_feature_engineering():
    df = pd.DataFrame({"Time":[0.,3600.], "Amount":[10.,100.], "V1":[.1,.2], "Class":[0,1]})
    X, y = split_features_target(df)
    assert "Hour" in X.columns
    assert "AmountLog" in X.columns
    assert y.tolist() == [0,1]
