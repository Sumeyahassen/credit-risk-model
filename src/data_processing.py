import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(raw_path='data/raw/transactions.csv'):
    """Load raw transaction data"""
    df = pd.read_csv(raw_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def create_rfm_features(df):
    """Create Recency, Frequency, Monetary (RFM) features"""
    
    # Snapshot date = one day after the latest transaction
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,   # Recency
        'TransactionId': 'count',                                           # Frequency
        'Value': ['sum', 'mean', 'std', 'max'],                             # Monetary (using Value = absolute)
        'Amount': 'mean'                                                    # Average signed amount
    })
    
    # Flatten multi-level columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary', 'AvgValue', 'StdValue', 'MaxValue', 'AvgAmount']
    rfm = rfm.reset_index()
    
    return rfm

def create_proxy_target(rfm_df, n_clusters=4):
    """Create is_high_risk target using KMeans clustering on RFM"""
    
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_summary = rfm_df.groupby('Cluster')[features].mean()
    print("=== Cluster Summary ===")
    print(cluster_summary.round(2))
    
    # High Risk = Cluster with highest Recency + lowest Frequency & Monetary
    # Usually the cluster with lowest Frequency
    high_risk_cluster = cluster_summary['Frequency'].idxmin()
    print(f"\n→ High Risk Cluster identified as: {high_risk_cluster}")
    
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm_df

def process_data():
    """Main function to process data end-to-end"""
    df = load_data()
    rfm_df = create_rfm_features(df)
    final_df = create_proxy_target(rfm_df)
    
    # Save processed data
    final_df.to_csv('data/processed/processed_data.csv', index=False)
    print(f"\n✅ Processed data saved with shape: {final_df.shape}")
    
    return final_df

# For testing
if __name__ == "__main__":
    process_data()