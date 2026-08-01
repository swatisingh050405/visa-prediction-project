import numpy as np
import pandas as pd
from us_visa.constants import CURRENT_YEAR

def perform_feature_engineering(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    df = df.copy()

    # 1. Company Age
    if 'yr_of_estab' in df.columns:
        df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
        df['company_age'] = df['company_age'].clip(lower=0)

    # 2. Wage vs Regional Median Ratio
    if 'region_of_employment' in df.columns and 'prevailing_wage' in df.columns:
        region_median = df.groupby('region_of_employment')['prevailing_wage'].transform('median')
        df['wage_to_region_ratio'] = df['prevailing_wage'] / (region_median + 1e-5)

    # 3. Wage vs Education Level Ratio (High Signal)
    if 'education_of_employee' in df.columns and 'prevailing_wage' in df.columns:
        edu_median = df.groupby('education_of_employee')['prevailing_wage'].transform('median')
        df['wage_to_edu_ratio'] = df['prevailing_wage'] / (edu_median + 1e-5)

    # 4. Log Scale Company Employees
    if 'no_of_employees' in df.columns:
        df['log_no_of_employees'] = np.log1p(df['no_of_employees'].clip(lower=0))

    # 5. Drop specified columns
    if drop_cols:
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=existing_drop_cols, axis=1)

    return df