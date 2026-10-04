import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class DataPreprocessor:
    def __init__(self, target_column: str = "Loan_Status", drop_columns: Optional[List[str]] = None):
        self.target_column = target_column
        self.drop_columns = drop_columns or ["Loan_ID", "id", "ID"]
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_out: List[str] = []
        self.pipeline: Optional[ColumnTransformer] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.is_fitted: bool = False
        self.summary_stats: Dict[str, Any] = {}

    def detect_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cols_to_check = [c for c in df.columns if c != self.target_column and c not in self.drop_columns]
        
        num_cols = []
        cat_cols = []
        
        for col in cols_to_check:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If unique values are just 0 and 1, e.g. Credit_History, check if categorical or numeric
                # In standard loan dataset, Credit_History (0.0/1.0) can be treated numerically or categorically.
                # Standard practice treats Credit_History as numeric binary or categorical with 0/1.
                # We'll treat Credit_History as numerical for scaling/imputation or categorical.
                # Let's treat Credit_History as numerical with median imputation (which gives 1.0).
                num_cols.append(col)
            else:
                cat_cols.append(col)
                
        self.numerical_cols = num_cols
        self.categorical_cols = cat_cols
        return num_cols, cat_cols

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics on dataset before preprocessing."""
        num_rows, num_cols = df.shape
        missing_dict = df.isnull().sum().to_dict()
        missing_pct = {k: round((v / num_rows) * 100, 2) for k, v in missing_dict.items()}
        duplicates_count = int(df.duplicated().sum())
        
        target_dist = {}
        if self.target_column in df.columns:
            target_counts = df[self.target_column].value_counts().to_dict()
            target_pct = df[self.target_column].value_counts(normalize=True).mul(100).round(2).to_dict()
            for k in target_counts:
                target_dist[str(k)] = {
                    "count": int(target_counts[k]),
                    "percentage": float(target_pct[k])
                }

        self.summary_stats = {
            "total_rows": num_rows,
            "total_columns": num_cols,
            "columns": list(df.columns),
            "missing_values": missing_dict,
            "missing_percentages": missing_pct,
            "duplicate_rows": duplicates_count,
            "target_column": self.target_column,
            "target_distribution": target_dist,
            "numerical_columns": self.numerical_cols,
            "categorical_columns": self.categorical_cols
        }
        return self.summary_stats

    def build_pipeline(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer to prevent data leakage."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if self.numerical_cols:
            transformers.append(('num', numeric_transformer, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', categorical_transformer, self.categorical_cols))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor strictly on training data and transform features and target."""
        self.detect_columns(df)
        self.inspect_dataset(df)
        self.build_pipeline()

        X = df.drop(columns=[c for c in [self.target_column] + self.drop_columns if c in df.columns])
        y_raw = df[self.target_column].copy()

        # Handle target encoding
        # Normalize target: 'Y', '1', 1, 'Approved' -> 1; 'N', '0', 0, 'Rejected' -> 0
        if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
            y_mapped = y_raw.astype(str).str.strip().str.upper().map(lambda v: 1 if v in ['Y', 'YES', 'APPROVED', '1'] else 0)
        else:
            y_mapped = (y_raw == 1).astype(int)
            
        y = y_mapped.values

        X_transformed = self.pipeline.fit_transform(X)
        self.is_fitted = True

        # Extract transformed feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(list(cat_feature_names))
            
        self.feature_names_out = feature_names
        return X_transformed, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform test data or new inputs without refitting (preventing leakage)."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet!")

        X = df.drop(columns=[c for c in [self.target_column] + self.drop_columns if c in df.columns], errors='ignore')
        return self.pipeline.transform(X)

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single application dict into scaled/encoded feature vector."""
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)

    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            "target_column": self.target_column,
            "drop_columns": self.drop_columns,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "feature_names_out": self.feature_names_out,
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
            "summary_stats": self.summary_stats
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor state from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        state = joblib.load(filepath)
        instance = cls(target_column=state["target_column"], drop_columns=state["drop_columns"])
        instance.numerical_cols = state["numerical_cols"]
        instance.categorical_cols = state["categorical_cols"]
        instance.feature_names_out = state["feature_names_out"]
        instance.pipeline = state["pipeline"]
        instance.is_fitted = state["is_fitted"]
        instance.summary_stats = state.get("summary_stats", {})
        return instance
