import numpy as np
import numpy.typing as npt
import pandas as pd
import copy
from sklearn import preprocessing # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report # type: ignore

class SalaryPredictor:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # Initialize OneHotEncoder for categorical features
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Initialize StandardScaler for normalizing numerical features
        self.scaler = StandardScaler()
        
        features = self.preprocess(X_train, True)
        self.lrbc = LogisticRegression(max_iter=5000)#trains a Logistic Regression Model up to 5000 iterations
        self.lrbc.fit(features, y_train)
        #x train is the training data
        #y train is the training labels

    def preprocess (self, features: pd.DataFrame, training: bool = False) -> npt.NDArray:

        
        # Make a copy to avoid modifying the original DataFrame
        features_clean = features.copy()
        
        # Clean up excess whitespace from string columns
        for col in features_clean.columns:
            if features_clean[col].dtype == 'object':  # String columns
                features_clean[col] = features_clean[col].str.strip()
        
        # Separate categorical and numerical columns
        categorical_cols = features_clean.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = features_clean.select_dtypes(exclude=['object']).columns.tolist()
        
        # Extract categorical and numerical data
        # Both extract same row order from features_clean
        categorical_data = features_clean[categorical_cols] if categorical_cols else pd.DataFrame()
        if numerical_cols:
            # Extract numerical columns keeping row order
            numerical_data = np.asarray(features_clean[numerical_cols].values)
        else:
            numerical_data = np.array([]).reshape(len(features_clean), 0)
        
        # Scale numerical features to handle varying scales
        # This ensures features with larger scales don't dominate the model
        if len(numerical_cols) > 0:
            if training:
                # Fit and transform during training
                numerical_scaled = self.scaler.fit_transform(numerical_data)
            else:
                # Only transform during testing (use statistics from training)
                numerical_scaled = self.scaler.transform(numerical_data)
        else:
            numerical_scaled = np.array([]).reshape(len(features_clean), 0)
        
        # One-hot encode categorical features keeping row order
        if len(categorical_cols) > 0:
            if training:
                # Fit and transform during training
                onehot_encoded = self.onehot_encoder.fit_transform(categorical_data)
            else:
                # Only transform during testing
                onehot_encoded = self.onehot_encoder.transform(categorical_data)
            
            # Ensure onehot_encoded is a numpy array
            onehot_encoded = np.asarray(onehot_encoded)
            
            # Combine scaled numerical and one-hot encoded features horizontally
            # This keeps each row together: row i from numerical_scaled 
            # stays with row i from onehot_encoded
            if len(numerical_cols) > 0:
                features_final = np.hstack([numerical_scaled, onehot_encoded])
            else:
                features_final = onehot_encoded
        else:
            # Only numerical features are scaled
            features_final = numerical_scaled
        
        # return type is a numpy array
        return np.asarray(features_final)

    def classify (self, X_test: pd.DataFrame) -> list[int]:
        return list(self.lrbc.predict(self.preprocess(X_test, False)))

    def test_model (self, X_test: "pd.DataFrame", y_test: "pd.DataFrame") -> tuple[str, dict]:
        prediction = self.classify(X_test)
        return (classification_report(y_test,prediction, output_dict = False),
                classification_report(y_test,prediction, output_dict = True))
        
