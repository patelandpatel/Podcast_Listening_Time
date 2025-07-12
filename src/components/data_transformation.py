import sys
from dataclasses import dataclass
import re
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at 95th percentile for specific columns"""
    
    def __init__(self, columns_to_cap):
        self.columns_to_cap = columns_to_cap
        self.caps_ = {}
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        for col_idx in self.columns_to_cap:
            if col_idx < df.shape[1]:  # Safety check
                self.caps_[col_idx] = np.percentile(df.iloc[:, col_idx].dropna(), 95)
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X)
        for col_idx, cap_value in self.caps_.items():
            if col_idx < df.shape[1]:  # Safety check
                df.iloc[:, col_idx] = np.clip(df.iloc[:, col_idx], None, cap_value)
        return df.values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features from Publication_Day and Publication_Time"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X, columns=['Publication_Day', 'Publication_Time'])
        
        # Create time-based features
        df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
        df['Is_Morning'] = (df['Publication_Time'] == 'Morning').astype(int)
        df['Is_Evening'] = df['Publication_Time'].isin(['Evening', 'Night']).astype(int)
        df['Weekend_Evening'] = (df['Is_Weekend'] & df['Is_Evening']).astype(int)
        
        # Return only the new features
        return df[['Is_Weekend', 'Is_Morning', 'Is_Evening', 'Weekend_Evening']].values


class EpisodeTitleProcessor(BaseEstimator, TransformerMixin):
    """Extract episode numbers and categorize them"""
    
    def __init__(self):
        self.percentile_33_ = None
        self.percentile_66_ = None
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=['Episode_Title'])
        
        # Extract episode numbers
        episode_numbers = df['Episode_Title'].apply(self._extract_episode_number)
        episode_numbers = episode_numbers.dropna()
        
        if len(episode_numbers) > 0:
            # Calculate percentiles
            self.percentile_33_ = np.percentile(episode_numbers, 33)
            self.percentile_66_ = np.percentile(episode_numbers, 66)
        else:
            # Fallback values if no episode numbers found
            self.percentile_33_ = 1
            self.percentile_66_ = 2
        
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X, columns=['Episode_Title'])
        
        # Extract episode numbers
        df['Episode_Number'] = df['Episode_Title'].apply(self._extract_episode_number)
        
        # Categorize episodes
        df['Episode_Category'] = df['Episode_Number'].apply(self._categorize_episode)
        
        return df[['Episode_Category']].values
    
    def _extract_episode_number(self, title):
        """Extract episode number from title like 'Episode 98'"""
        if pd.isna(title):
            return np.nan
        
        match = re.search(r'Episode\s+(\d+)', str(title), re.IGNORECASE)
        if match:
            return int(match.group(1))
        return np.nan
    
    def _categorize_episode(self, episode_num):
        """Categorize episode number into Beg/Med/Advanced"""
        if pd.isna(episode_num):
            return 'Unknown'
        
        if episode_num <= self.percentile_33_:
            return 'Beg'
        elif episode_num <= self.percentile_66_:
            return 'Med'
        else:
            return 'Advanced'


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """K-Fold Target Encoder for high cardinality categorical features"""
    
    def __init__(self, n_splits=5, random_state=42, smoothing=1.0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None
        
    def fit(self, X, y):
        df = pd.DataFrame(X)
        self.global_mean_ = np.mean(y)
        
        # Use K-Fold cross-validation to prevent overfitting
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for col_idx in range(df.shape[1]):
            col_name = f'col_{col_idx}'
            encoded_values = np.zeros(len(df))
            
            for train_idx, val_idx in kf.split(df):
                # Calculate target means for training fold
                train_df = df.iloc[train_idx]
                train_y = y[train_idx]
                
                fold_means = {}
                for category in train_df.iloc[:, col_idx].unique():
                    if pd.notna(category):
                        mask = train_df.iloc[:, col_idx] == category
                        if np.sum(mask) > 0:  # Avoid empty groups
                            category_mean = np.mean(train_y[mask])
                            # Apply smoothing
                            category_count = np.sum(mask)
                            smoothed_mean = (category_mean * category_count + self.global_mean_ * self.smoothing) / (category_count + self.smoothing)
                            fold_means[category] = smoothed_mean
                
                # Encode validation fold
                for idx in val_idx:
                    category = df.iloc[idx, col_idx]
                    encoded_values[idx] = fold_means.get(category, self.global_mean_)
            
            # Store final mappings for transform
            category_means = {}
            for category in df.iloc[:, col_idx].unique():
                if pd.notna(category):
                    mask = df.iloc[:, col_idx] == category
                    if np.sum(mask) > 0:  # Avoid empty groups
                        category_means[category] = np.mean(y[mask])
            
            self.target_means_[col_idx] = category_means
        
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X)
        encoded_df = df.copy()
        
        for col_idx in range(df.shape[1]):
            if col_idx in self.target_means_:
                encoded_df.iloc[:, col_idx] = df.iloc[:, col_idx].map(
                    self.target_means_[col_idx]
                ).fillna(self.global_mean_)
        
        return encoded_df.values


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocess_missing_values(self, df):
        """
        Handle all missing values FIRST before main pipeline
        Sequential processing to avoid parallel conflicts
        """
        try:
            df_processed = df.copy()
            
            logging.info("Starting missing value preprocessing...")
            
            # 1. Fill Guest_Popularity_percentage with 0 (no guest)
            df_processed['Guest_Popularity_percentage'].fillna(0, inplace=True)
            logging.info(" Guest_Popularity_percentage missing values filled with 0")
            
            # 2. Fill Number_of_Ads with median
            median_ads = df_processed['Number_of_Ads'].median()
            df_processed['Number_of_Ads'].fillna(median_ads, inplace=True)
            logging.info(f" Number_of_Ads missing values filled with median: {median_ads}")
            
            # 3. Fill Episode_Length_minutes with genre-based mean
            # First, handle missing genres
            df_processed['Genre'].fillna('Unknown', inplace=True)
            
            # Calculate genre means for Episode_Length_minutes
            genre_means = df_processed.groupby('Genre')['Episode_Length_minutes'].mean()
            
            for genre in df_processed['Genre'].unique():
                if pd.notna(genre) and genre in genre_means and not pd.isna(genre_means[genre]):
                    mask = (df_processed['Genre'] == genre) & (df_processed['Episode_Length_minutes'].isna())
                    df_processed.loc[mask, 'Episode_Length_minutes'] = genre_means[genre]
            
            # Fill any remaining Episode_Length_minutes with overall mean
            overall_mean = df_processed['Episode_Length_minutes'].mean()
            df_processed['Episode_Length_minutes'].fillna(overall_mean, inplace=True)
            logging.info(" Episode_Length_minutes missing values filled with genre-based means")
            
            # 4. Fill any remaining missing values in categorical columns
            categorical_columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 
                                 'Publication_Time', 'Episode_Sentiment']
            
            for col in categorical_columns:
                if col in df_processed.columns:
                    df_processed[col].fillna('Unknown', inplace=True)
            
            logging.info(" All missing values handled successfully")
            logging.info(f"Remaining missing values: {df_processed.isnull().sum().sum()}")
            
            return df_processed
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        This function creates the complete preprocessing pipeline
        Now working with clean data (no missing values)
        """
        try:
            # Define column groups clearly - NO missing values at this point
            numerical_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
            host_popularity_column = ['Host_Popularity_percentage']
            time_columns = ['Publication_Day', 'Publication_Time']
            episode_title_columns = ['Episode_Title']
            
            # Columns for target encoding (high cardinality)
            target_encoding_columns = ['Podcast_Name', 'Genre']
            
            # Columns for one-hot encoding (low cardinality)  
            onehot_columns = ['Episode_Sentiment']

            # Create preprocessing pipelines
            
            # 1. Numerical pipeline (outlier capping + scaling)
            numerical_pipeline = Pipeline([
                ('outlier_capper', OutlierCapper([0, 1, 2])),  # All three numerical columns
                ('scaler', StandardScaler())
            ])
            
            # 2. Host popularity pipeline (outlier capping + scaling)
            host_pipeline = Pipeline([
                ('outlier_capper', OutlierCapper([0])),  # Only one column
                ('scaler', StandardScaler())
            ])
            
            # 3. Feature engineering pipeline for time features
            feature_engineering_pipeline = Pipeline([
                ('feature_engineer', FeatureEngineer())
            ])
            
            # 4. Episode title processing pipeline
            episode_pipeline = Pipeline([
                ('episode_processor', EpisodeTitleProcessor()),
                ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # 5. Target encoding pipeline for high cardinality categorical features
            target_encoding_pipeline = Pipeline([
                ('target_encoder', KFoldTargetEncoder()),
                ('scaler', StandardScaler())
            ])
            
            # 6. One-hot encoding pipeline for low cardinality categorical features
            onehot_pipeline = Pipeline([
                ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Combine all pipelines
            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, numerical_columns),
                ('host_popularity', host_pipeline, host_popularity_column),
                ('time_features', feature_engineering_pipeline, time_columns),
                ('episode_features', episode_pipeline, episode_title_columns),
                ('target_encoding', target_encoding_pipeline, target_encoding_columns),
                ('onehot_encoding', onehot_pipeline, onehot_columns)
            ], remainder='drop')  # Drop any remaining columns

            logging.info("Preprocessing pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Check for missing values before preprocessing
            logging.info("Missing values in training data BEFORE preprocessing:")
            logging.info(train_df.isnull().sum())

            # STEP 1: Handle missing values FIRST (Sequential preprocessing)
            logging.info("="*50)
            logging.info("STEP 1: Handling missing values sequentially...")
            train_df_clean = self.preprocess_missing_values(train_df)
            test_df_clean = self.preprocess_missing_values(test_df)

            # Verify no missing values remain
            logging.info("Missing values in training data AFTER preprocessing:")
            logging.info(train_df_clean.isnull().sum())

            # STEP 2: Apply main pipeline (No missing value conflicts)
            logging.info("="*50)
            logging.info("STEP 2: Applying main preprocessing pipeline...")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Listening_Time_minutes"

            # Separate features and target
            input_feature_train_df = train_df_clean.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df_clean[target_column_name]

            input_feature_test_df = test_df_clean.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df_clean[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")

            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df, target_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Transformed train shape: {train_arr.shape}")
            logging.info(f"Transformed test shape: {test_arr.shape}")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")
            logging.info("="*50)
            logging.info(" Data transformation completed successfully!")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)