import numpy as np
import pandas as pd
import os
import json
import pickle
import warnings

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler


class MyARIMA:
    def __init__(self, args=None):
        if args:
            self.p = getattr(args, 'p', 1)  # Auto-regressive order
            self.d = getattr(args, 'd', 1)  # Differencing order
            self.q = getattr(args, 'q', 0)  # Moving average order
            self.time_steps = args.time_steps  # For compatibility
            self.target_horizon = args.target_horizon  # How many hours ahead to predict
        else:
            # Default values if loading from file
            self.p = 1
            self.d = 1
            self.q = 0
            self.time_steps = None
            self.target_horizon = None
            
        self.models = {}  # Dictionary to store one ARIMA model per target variable
        self.is_model_created = False
        self.sc_in = MinMaxScaler(feature_range=(0, 1))
        self.sc_out = MinMaxScaler(feature_range=(0, 1))
        
        # Define OHLCV columns (these will be the target for prediction)
        self.ohlcv_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume_to']
        self.num_output_features = len(self.ohlcv_columns)

    def create_model(self):
        """Create separate ARIMA models for each target variable"""
        # ARIMA models will be created during fitting
        # We just initialize an empty dictionary here
        self.models = {col: None for col in self.ohlcv_columns}
        self.is_model_created = True
        return self.models

    def prepare_data(self, df):
        """
        Prepare data for training - separate features and targets
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features and target columns
            
        Returns:
        --------
        features_df : DataFrame
            DataFrame containing only the feature columns
        targets_df : DataFrame 
            DataFrame containing only the target columns (OHLCV)
        """
        # Extract target columns for the next day prediction
        features_df = df.drop(columns=['datetime'] if 'datetime' in df.columns else [])
        
        # Get target columns (which are the OHLCV columns for the next time period)
        targets_df = df[self.ohlcv_columns]
        
        return features_df, targets_df

    def fit(self, df):
        """Train the ARIMA models using a DataFrame
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features including OHLCV data
        """
        # Prepare targets only - ARIMA is a univariate method
        _, targets_df = self.prepare_data(df)
        
        # We don't need to scale for ARIMA, but we'll fit the scalers for prediction compatibility
        self.sc_in.fit(df)
        self.sc_out.fit(targets_df)
        
        # Create model instances if not already done
        if not self.is_model_created:
            self.create_model()
        
        # Train a separate ARIMA model for each target column
        for col in self.ohlcv_columns:
            # Suppress convergence warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Fit ARIMA model
                self.models[col] = ARIMA(targets_df[col], order=(self.p, self.d, self.q))
                self.models[col] = self.models[col].fit()
        
        # Return a basic history-like object for compatibility
        history = {'info': 'ARIMA models fitted'}
        
        return history

    def predict(self, df):
        """Make predictions with the ARIMA models
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features
            
        Returns:
        --------
        predictions : DataFrame
            DataFrame containing the predicted OHLCV values
        """
        # Check if we have models
        if not self.is_model_created or not self.models:
            raise ValueError("Models must be trained before prediction")
            
        # For ARIMA to work properly, we should use the original training data
        # followed by the data we want to predict. We'll assume df includes
        # the most recent observations needed for prediction.
        
        # Number of steps to predict
        steps = 1
        
        # Make predictions for each target column
        predictions = {}
        for col in self.ohlcv_columns:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Get the model for this column
                model = self.models[col]
                
                # Forecast
                forecast = model.forecast(steps=steps)
                
                # Store the prediction
                predictions[col] = forecast.values
        
        # Convert to DataFrame
        pred_array = np.column_stack([predictions[col] for col in self.ohlcv_columns])
        pred_df = pd.DataFrame(
            pred_array, 
            columns=self.ohlcv_columns,
            index=df.index[-steps:]  # Use the last index
        )
        
        return pred_df
    
    def predict_future(self, df, steps_ahead=24):
        """Predict multiple steps into the future
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features
        steps_ahead : int
            Number of steps to predict into the future
            
        Returns:
        --------
        future_predictions : DataFrame
            DataFrame containing the predicted OHLCV values for future steps
        """
        # Check if we have models
        if not self.is_model_created or not self.models:
            raise ValueError("Models must be trained before prediction")
        
        # Make predictions for each target column
        predictions = {}
        for col in self.ohlcv_columns:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Get the model for this column
                model = self.models[col]
                
                # Forecast all steps at once
                forecast = model.forecast(steps=steps_ahead)
                
                # Store the predictions
                predictions[col] = forecast.values
        
        # Convert to DataFrame
        future_predictions = []
        for i in range(steps_ahead):
            row = {col: predictions[col][i] for col in self.ohlcv_columns}
            future_predictions.append(row)
        
        future_df = pd.DataFrame(future_predictions, columns=self.ohlcv_columns)
        
        return future_df
    
    def save(self, folder_path):
        """Save the model and associated objects to a folder"""
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Save ARIMA model results - we need to save differently than Keras models
        for col in self.ohlcv_columns:
            if self.models[col] is not None:
                # Save model results as pickle
                with open(os.path.join(folder_path, f'arima_model_{col}.pkl'), 'wb') as f:
                    pickle.dump(self.models[col], f)
            
        # Save scalers for compatibility
        with open(os.path.join(folder_path, 'sc_in.pkl'), 'wb') as f:
            pickle.dump(self.sc_in, f)
            
        with open(os.path.join(folder_path, 'sc_out.pkl'), 'wb') as f:
            pickle.dump(self.sc_out, f)
            
        # Save hyperparameters
        params = {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'time_steps': self.time_steps,
            'target_horizon': self.target_horizon,
            'is_model_created': self.is_model_created,
            'ohlcv_columns': self.ohlcv_columns,
            'num_output_features': self.num_output_features
        }
        
        with open(os.path.join(folder_path, 'params.json'), 'w') as f:
            json.dump(params, f)
            
        print(f"Model saved to {folder_path}")
    
    @classmethod
    def load(cls, folder_path):
        """Load the model and associated objects from a folder"""
        # Create a new instance
        instance = cls()
        
        # Load hyperparameters
        with open(os.path.join(folder_path, 'params.json'), 'r') as f:
            params = json.load(f)
            
        instance.p = params['p']
        instance.d = params['d']
        instance.q = params['q']
        instance.time_steps = params['time_steps']
        instance.target_horizon = params['target_horizon']
        instance.is_model_created = params['is_model_created']
        instance.ohlcv_columns = params['ohlcv_columns']
        instance.num_output_features = params['num_output_features']
        
        # Initialize the model dictionary
        instance.models = {}
        
        # Load each ARIMA model
        for col in instance.ohlcv_columns:
            model_path = os.path.join(folder_path, f'arima_model_{col}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    instance.models[col] = pickle.load(f)
        
        # Load scalers for compatibility
        with open(os.path.join(folder_path, 'sc_in.pkl'), 'rb') as f:
            instance.sc_in = pickle.load(f)
            
        with open(os.path.join(folder_path, 'sc_out.pkl'), 'rb') as f:
            instance.sc_out = pickle.load(f)
            
        print(f"Model loaded from {folder_path}")
        return instance

    # Additional helper method for ARIMA
    def update_model(self, new_observation):
        """Update the ARIMA models with new observations
        
        Parameters:
        -----------
        new_observation : DataFrame
            DataFrame containing new observations of OHLCV data
        """
        # Make sure new_observation contains all target columns
        for col in self.ohlcv_columns:
            if col not in new_observation.columns:
                raise ValueError(f"New observation missing column: {col}")
        
        # Update each model
        for col in self.ohlcv_columns:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Get the model for this column
                model = self.models[col]
                
                # Update with new observation
                model = model.append(new_observation[col])
                
                # Store updated model
                self.models[col] = model
                
        return self