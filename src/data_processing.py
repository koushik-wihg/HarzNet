import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.common import read_params
from src.utils.transforms import clr_transform
import joblib
import numpy as np

def get_preprocessor(params, numerical_cols):
    impute_strategy = params['data_processing']['impute_strategy']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('clr', FunctionTransformer(clr_transform, validate=False)),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols)
    ], remainder='drop')

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)

    target_col = params['base']['target_column']
    raw_data_path = Path(params['data_ingestion']['output_file'])
    passthrough = params['data_processing']['passthrough_features']
    pca_var = params['data_processing']['pca_variance_threshold']

    processed_data_path = Path("data/processed/data_processed.pkl")
    pipeline_path = Path("models/preprocessing_pipeline.joblib")

    try:
        data = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print("Error: Raw data not found. Run data_ingestion first.")
        return

    X = data.drop(columns=[target_col], errors='ignore')
    y = data[target_col]

    num_cols = [c for c in X.columns if X[c].dtype in ['int64','float64'] and c not in passthrough]
    if not num_cols:
        print("Error: No numerical columns found for preprocessing/PCA.")
        return

    preprocessor = get_preprocessor(params, num_cols)

    X_trans = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X_trans, columns=num_cols)

    pca = PCA(n_components=pca_var, random_state=params['data_ingestion']['random_state'])
    pca_df = pd.DataFrame(
        pca.fit_transform(X_df),
        columns=[f'PCA_{i+1}' for i in range(pca.n_components_)]
    )

    final = pd.concat([
        pca_df,
        X[passthrough].reset_index(drop=True),
        y.reset_index(drop=True)
    ], axis=1)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_pickle(processed_data_path)

    joblib.dump(
        Pipeline([("pre", preprocessor), ("pca", pca)]),
        pipeline_path
    )

    print(f"Data processed and saved. PCA components found: {pca.n_components_}")

if __name__ == "__main__":
    main()
