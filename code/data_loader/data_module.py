import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

BATCH_SIZE = 16 # Keep batch size consistent with nn_example for now

# Determine base directory dynamically
CODEOCEAN_BASE_DIR = os.environ.get('CODEOCEAN_BASE_DIR', '/')
DATA_PATH = os.path.join(CODEOCEAN_BASE_DIR, 'data')

def load_iris_data():
    filepath = os.path.join(DATA_PATH, 'Iris.csv')
    df = pd.read_csv(filepath)

    # Map species names to numerical labels
    species_map = {name: i for i, name in enumerate(df['Species'].unique())}
    df['Species'] = df['Species'].map(species_map)

    # Define features (X) and target (y)
    # Drop 'Id' column as it's not a feature
    X = df.drop(['Id', 'Species'], axis=1).values
    y = df['Species'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Return target names based on the mapped species
    target_names = [name for name, i in sorted(species_map.items(), key=lambda item: item[1])]
    return trainloader, testloader, target_names, X_train.shape[1], len(target_names)
