import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Unexpected error loading params: %s', e)
        raise

def main():
    try:
        # 1. Load parameters for feature engineering
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        # 2. Read the preprocessed clean data from data/interim
        logger.debug("Loading interim datasets from data/interim...")
        train_df = pd.read_csv(os.path.join('data/interim', 'train.csv'))
        test_df = pd.read_csv(os.path.join('data/interim', 'test.csv'))
        
        # Ensure text columns don't have nulls (common culprit with tweet data)
        train_df['content'] = train_df['content'].fillna('')
        test_df['content'] = test_df['content'].fillna('')

        # 3. Fit the Vectorizer
        logger.debug("Applying TF-IDF vectorization with max_features=%d...", max_features)
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = vectorizer.fit_transform(train_df['content']).toarray()
        X_test = vectorizer.transform(test_df['content']).toarray()
        
        y_train = train_df['sentiment'].values
        y_test = test_df['sentiment'].values

        # 4. Package up matrices to save them out safely
        # Combining X and y so they can be easily loaded in the model building stage
        train_features = pd.DataFrame(X_train)
        train_features['label'] = y_train
        
        test_features = pd.DataFrame(X_test)
        test_features['label'] = y_test

        # 5. Build outputs exactly where dvc.yaml expects them
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        train_features.to_csv(os.path.join('data/processed', 'train_features.csv'), index=False)
        test_features.to_csv(os.path.join('data/processed', 'test_features.csv'), index=False)
        logger.debug('Processed features saved to data/processed')
        
        # 6. Save the missing vectorizer object that caused the DVC error!
        with open(os.path.join('models', 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug('Vectorizer saved successfully to models/vectorizer.pkl')

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()