import numpy as np
import pandas as pd
import os
import json
import pickle

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor


class MyXGBoost:
    def __init__(self, args=None):
        if args:
            self.n_estimators = getattr(args, 'n_estimators', 100)
            self.max_depth = getattr(args, 'max_depth', 3)
            self.learning_rate = getattr(args, 'learning_rate', 0.1)
            self.time_steps = args.time_steps  # Number of time steps (previous days) to use
            self.target_horizon = args.target_horizon  # How many hours ahead to predict
        else:
            # Default values if loading from file
            self.n_estimators = 100
            self.max_depth = 3
            self.learning_rate = 0.1
            self.time_steps = None
            self.target_horizon = None
            
        self.model = None
        self.is_model_created = False
        self.sc_in = MinMaxScaler(feature_range=(0, 1))
        self.sc_out = MinMaxScaler(feature_range=(0, 1))
        
        # Define OHLCV columns (these will be the target for prediction)
        self.ohlcv_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume_to']
        self.num_output_features = len(self.ohlcv_columns)
        
        # Feature importance tracking
        self.feature_importances_ = None
        self.feature_names_ = None

    def create_model(self):
        """Create the XGBoost model"""
        # Create base model with specified hyperparameters
        base_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            n_jobs=-1,  # Use all available CPUs
            random_state=42
        )
        
        # Wrap with MultiOutputRegressor to handle multiple target variables
        self.model = MultiOutputRegressor(base_model)
        self.is_model_created = True
        return self.model

    def create_sequences(self, data, target_data=None):
        """Create sequences for time series prediction
        
        Parameters:
        -----------
        data : DataFrame or ndarray
            Input features to create sequences from
        target_data : DataFrame or ndarray, optional
            Target values (if None, assumes no targets available for prediction mode)
            
        Returns:
        --------
        X : ndarray
            Flattened time series features with shape (samples, time_steps * features)
        y : ndarray or None
            Target values with shape (samples, num_targets) or None if target_data is None
        """
        # Convert data to numpy array if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if target_data is not None and isinstance(target_data, pd.DataFrame):
            target_data = target_data.values
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(data) - self.time_steps):
            # Take time_steps worth of data and flatten for XGBoost (can't handle 3D data)
            X.append(data[i:i + self.time_steps].flatten())
            
            # If we have target data, include the corresponding target
            if target_data is not None:
                y.append(target_data[i + self.time_steps])
        
        X = np.array(X)
        
        if target_data is not None:
            y = np.array(y)
        else:
            y = None
            
        return X, y

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
        """Train the model using a DataFrame with time windows
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features including OHLCV data
        """
        # Prepare features and targets
        features_df, targets_df = self.prepare_data(df)
        
        # Store feature names for later feature importance extraction
        self.feature_names_ = features_df.columns.tolist()
        
        # Scale the data
        features_scaled = self.sc_in.fit_transform(features_df)
        targets_scaled = self.sc_out.fit_transform(targets_df)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, targets_scaled)
        
        # Create model if not already done
        if not self.is_model_created:
            self.create_model()
        
        # Train the model
        self.model.fit(X, y)
        
        # Extract feature importances from all estimators and average them
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        # Average feature importances across all targets (OHLCV)
        self.feature_importances_ = np.mean(importances, axis=0)
        
        # Return a basic history-like object for compatibility
        history = {'n_estimators': self.n_estimators}
        
        return history

    def predict(self, df):
        """Make predictions with the model
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with all features
            
        Returns:
        --------
        predictions : DataFrame
            DataFrame containing the predicted OHLCV values
        """
        # Prepare features
        features_df, _ = self.prepare_data(df)
        
        # Scale features
        features_scaled = self.sc_in.transform(features_df)
        
        # Create sequences (without targets)
        X, _ = self.create_sequences(features_scaled)
        
        # Make predictions
        pred_scaled = self.model.predict(X)
        
        # Inverse transform to get original scale
        predictions = self.sc_out.inverse_transform(pred_scaled)
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(
            predictions, 
            columns=self.ohlcv_columns,
            index=df.index[self.time_steps:]  # Align indices correctly
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
        # Start with the most recent available data
        current_df = df.copy()
        future_predictions = []
        
        for step in range(steps_ahead):
            # Make a single prediction
            next_pred = self.predict(current_df)
            future_predictions.append(next_pred.iloc[-1])  # Take the last prediction
            
            # Create a new row with the predicted values
            new_row = current_df.iloc[-1].copy()
            
            # Update the OHLCV values in the new row
            for col in self.ohlcv_columns:
                new_row[col] = next_pred.iloc[-1][col]
                
            # Append the new row to current_df
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
            
        # Convert predictions to DataFrame
        future_df = pd.DataFrame(future_predictions, columns=self.ohlcv_columns)
        
        return future_df
    
    def get_feature_importance(self, top_n=None):
        """Get feature importance from the trained model
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return. If None, returns all features.
            
        Returns:
        --------
        feature_importance : DataFrame
            DataFrame containing feature names and importance scores
        """
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise ValueError("Model must be trained before feature importance can be extracted")
        
        # Repeat feature names for each time step
        time_step_features = []
        for t in range(self.time_steps):
            for feature in self.feature_names_:
                time_step_features.append(f"{feature}(t-{t})")
        
        # Create DataFrame with features and their importance
        feature_importance = pd.DataFrame({
            'Feature': time_step_features,
            'Importance': self.feature_importances_
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Limit to top_n features if specified
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)
            
        return feature_importance
    
    def save(self, folder_path):
        """Save the model and associated objects to a folder"""
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Save XGBoost model using pickle
        with open(os.path.join(folder_path, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save scalers using pickle
        with open(os.path.join(folder_path, 'sc_in.pkl'), 'wb') as f:
            pickle.dump(self.sc_in, f)
            
        with open(os.path.join(folder_path, 'sc_out.pkl'), 'wb') as f:
            pickle.dump(self.sc_out, f)
            
        # Save feature importances
        if self.feature_importances_ is not None and self.feature_names_ is not None:
            np.save(os.path.join(folder_path, 'feature_importances.npy'), self.feature_importances_)
            with open(os.path.join(folder_path, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_names_, f)
            
        # Save hyperparameters
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
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
            
        instance.n_estimators = params['n_estimators']
        instance.max_depth = params['max_depth']
        instance.learning_rate = params['learning_rate']
        instance.time_steps = params['time_steps']
        instance.target_horizon = params['target_horizon']
        instance.is_model_created = params['is_model_created']
        instance.ohlcv_columns = params['ohlcv_columns']
        instance.num_output_features = params['num_output_features']
        
        # Load XGBoost model
        with open(os.path.join(folder_path, 'xgb_model.pkl'), 'rb') as f:
            instance.model = pickle.load(f)
        
        # Load scalers
        with open(os.path.join(folder_path, 'sc_in.pkl'), 'rb') as f:
            instance.sc_in = pickle.load(f)
            
        with open(os.path.join(folder_path, 'sc_out.pkl'), 'rb') as f:
            instance.sc_out = pickle.load(f)
            
        # Load feature importances if available
        feature_importances_path = os.path.join(folder_path, 'feature_importances.npy')
        feature_names_path = os.path.join(folder_path, 'feature_names.json')
        
        if os.path.exists(feature_importances_path) and os.path.exists(feature_names_path):
            instance.feature_importances_ = np.load(feature_importances_path)
            with open(feature_names_path, 'r') as f:
                instance.feature_names_ = json.load(f)
            
        print(f"Model loaded from {folder_path}")
        return instance