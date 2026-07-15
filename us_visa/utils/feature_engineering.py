import pandas as pd

from us_visa.constants import CURRENT_YEAR
from us_visa.utils.main_utils import drop_columns


def perform_feature_engineering(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Performs all feature engineering required before preprocessing.
    """

    df = df.copy()

    # Create company age feature
    df["company_age"] = CURRENT_YEAR - df["yr_of_estab"]

    # Drop unwanted columns
    df = drop_columns(df=df, cols=drop_cols)

    return df