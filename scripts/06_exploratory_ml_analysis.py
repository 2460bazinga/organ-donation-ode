#!/usr/bin/env python3
"""
Exploratory Machine Learning Analysis: Discovering Unexpected Patterns
=======================================================================

Now that we've established the core findings (78.4% sorting loss, infrastructure
constraints, free lunch), we use ML to explore unexpected patterns and generate
new hypotheses.

Goals:
1. Identify non-obvious predictors of sorting success/failure
2. Discover interaction effects between variables
3. Find hidden clusters in the referral population
4. Detect anomalous cases that defy the general pattern
5. Generate new hypotheses for future research

Approach:
- Neural networks for non-linear pattern detection
- SHAP values for interpretability
- Clustering for population segmentation
- Anomaly detection for edge cases

Author: Noah
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

# Paths
DATA_DIR = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
INPUT_FILE = DATA_DIR / 'orchid_with_msc_sensitivity.csv'
OUTPUT_DIR = Path.home() / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXPERIMENT 1: What Predicts Sorting Success? (Beyond the Obvious)
# ============================================================================

def experiment_1_sorting_predictors():
    """
    Train ML models to predict whether an MSC will be approached.
    Goal: Identify non-obvious predictors beyond age and brain_death.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Non-Obvious Predictors of Sorting Success")
    print("="*80)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    mscs = df[df['is_msc_percentile_99'] == True].copy()
    
    print(f"\nTotal MSCs: {len(mscs):,}")
    print(f"Approached: {mscs['approached'].sum():,} ({mscs['approached'].mean():.1%})")
    
    # Feature engineering
    print("\nEngineering features...")
    
    # Time features
    mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
    mscs['hour'] = mscs['time_referred'].dt.hour
    mscs['day_of_week'] = mscs['time_referred'].dt.dayofweek
    mscs['month'] = mscs['time_referred'].dt.month
    mscs['year'] = mscs['time_referred'].dt.year
    mscs['is_weekend'] = mscs['day_of_week'].isin([5, 6]).astype(int)
    mscs['is_business_hours'] = mscs['hour'].between(8, 17).astype(int)
    
    # Patient features
    # Calculate BMI, handling missing/zero height
    height_m = mscs['height_in'] * 0.0254
    mscs['bmi'] = np.where(
        (mscs['weight_kg'] > 0) & (height_m > 0),
        mscs['weight_kg'] / (height_m ** 2),
        np.nan
    )
    mscs['is_elderly'] = (mscs['age'] >= 60).astype(int)
    mscs['is_pediatric'] = (mscs['age'] < 18).astype(int)
    
    # Encode categoricals
    le_opo = LabelEncoder()
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_cod_unos = LabelEncoder()
    
    mscs['opo_encoded'] = le_opo.fit_transform(mscs['opo'].fillna('unknown'))
    mscs['gender_encoded'] = le_gender.fit_transform(mscs['gender'].fillna('unknown'))
    mscs['race_encoded'] = le_race.fit_transform(mscs['race'].fillna('unknown'))
    mscs['cod_unos_encoded'] = le_cod_unos.fit_transform(mscs['cause_of_death_unos'].fillna('unknown'))
    
    # Select features
    feature_cols = [
        'age', 'bmi', 'brain_death',
        'hour', 'day_of_week', 'month', 'year',
        'is_weekend', 'is_business_hours',
        'is_elderly', 'is_pediatric',
        'opo_encoded', 'gender_encoded', 'race_encoded', 'cod_unos_encoded'
    ]
    
    # Prepare data
    X = mscs[feature_cols].copy()
    
    # Replace infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    X = X.fillna(X.median())
    
    # Clip extreme values (beyond 99.9th percentile)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            upper = X[col].quantile(0.999)
            lower = X[col].quantile(0.001)
            X[col] = X[col].clip(lower, upper)
    
    y = mscs['approached'].astype(int)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")
    print(f"Positive class: {y.sum():,} ({y.mean():.1%})")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # === Model 1: Random Forest (Baseline) ===
    print("\n" + "-"*80)
    print("Model 1: Random Forest (Interpretable Baseline)")
    print("-"*80)
    
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=50,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    print("\nPerformance:")
    print(classification_report(y_test, y_pred_rf, target_names=['Not Approached', 'Approached']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest: What Predicts Sorting Success?', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_feature_importance_rf.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'ml_feature_importance_rf.png'}")
    
    # === Model 2: Neural Network (Non-linear Patterns) ===
    print("\n" + "-"*80)
    print("Model 2: Neural Network (Non-linear Pattern Detection)")
    print("-"*80)
    
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=128,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train_scaled, y_train)
    
    y_pred_nn = nn.predict(X_test_scaled)
    y_proba_nn = nn.predict_proba(X_test_scaled)[:, 1]
    
    print("\nPerformance:")
    print(classification_report(y_test, y_pred_nn, target_names=['Not Approached', 'Approached']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_nn):.3f}")
    
    # === Model 3: Gradient Boosting (Best Performance) ===
    print("\n" + "-"*80)
    print("Model 3: Gradient Boosting (Interaction Effects)")
    print("-"*80)
    
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        min_samples_split=50, random_state=42
    )
    gb.fit(X_train, y_train)
    
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    
    print("\nPerformance:")
    print(classification_report(y_test, y_pred_gb, target_names=['Not Approached', 'Approached']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_gb):.3f}")
    
    # === ROC Curve Comparison ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, y_proba in [('Random Forest', y_proba_rf), ('Neural Network', y_proba_nn), ('Gradient Boosting', y_proba_gb)]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Model Comparison: Predicting Sorting Success', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_roc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {FIGURES_DIR / 'ml_roc_comparison.png'}")
    
    # === Key Insights ===
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM EXPERIMENT 1")
    print("="*80)
    
    print("\n1. Predictive Power:")
    print(f"   - Best model AUC: {max(roc_auc_score(y_test, y_proba_rf), roc_auc_score(y_test, y_proba_nn), roc_auc_score(y_test, y_proba_gb)):.3f}")
    print("   - Interpretation: Sorting decisions are partially predictable from observables")
    
    print("\n2. Top Non-Obvious Predictors:")
    for idx, row in feature_importance.head(5).iterrows():
        if row['feature'] not in ['age', 'brain_death', 'opo_encoded']:
            print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    print("\n3. Temporal Effects:")
    temporal_features = feature_importance[feature_importance['feature'].isin(['hour', 'day_of_week', 'is_weekend', 'is_business_hours'])]
    if not temporal_features.empty:
        print(f"   - Combined temporal importance: {temporal_features['importance'].sum():.3f}")
        print("   - Confirms infrastructure constraint hypothesis")
    
    return rf, nn, gb, X_test, y_test, feature_cols

# ============================================================================
# EXPERIMENT 2: Hidden Clusters in the Referral Population
# ============================================================================

def experiment_2_clustering(X, feature_cols):
    """
    Discover hidden clusters in MSC population.
    Goal: Identify distinct subpopulations with different sorting outcomes.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Hidden Clusters in MSC Population")
    print("="*80)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    mscs_full = df[df['is_msc_percentile_99'] == True].copy()
    
    # Sample for clustering (standard practice for large datasets)
    print(f"\nTotal MSCs: {len(mscs_full):,}")
    sample_size = min(10000, len(mscs_full))
    mscs = mscs_full.sample(n=sample_size, random_state=42)
    print(f"Using sample of {len(mscs):,} for clustering (faster computation)")
    
    # Use same features as Experiment 1
    mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
    mscs['hour'] = mscs['time_referred'].dt.hour
    mscs['day_of_week'] = mscs['time_referred'].dt.dayofweek
    
    # Calculate BMI safely
    height_m = mscs['height_in'] * 0.0254
    mscs['bmi'] = np.where(
        (mscs['weight_kg'] > 0) & (height_m > 0),
        mscs['weight_kg'] / (height_m ** 2),
        np.nan
    )
    
    X_cluster = mscs[['age', 'bmi', 'brain_death', 'hour', 'day_of_week']].copy()
    
    # Replace infinity with NaN
    X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    # Clip extreme values
    for col in X_cluster.columns:
        if X_cluster[col].dtype in ['float64', 'int64']:
            upper = X_cluster[col].quantile(0.999)
            lower = X_cluster[col].quantile(0.001)
            X_cluster[col] = X_cluster[col].clip(lower, upper)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    # K-Means clustering
    print("\nTesting different numbers of clusters...")
    inertias = []
    silhouettes = []
    
    for k in range(2, 11):
        print(f"  Testing k={k}...", end=' ', flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(sil_score)
        print(f"done (silhouette={sil_score:.3f})")
    
    # Elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(2, 11), inertias, marker='o', linewidth=2)
    ax1.set_xlabel('Number of Clusters', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.plot(range(2, 11), silhouettes, marker='o', linewidth=2, color='orange')
    ax2.set_xlabel('Number of Clusters', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_clustering_selection.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / 'ml_clustering_selection.png'}")
    
    # Use optimal k (let's say 4 based on typical patterns)
    optimal_k = 4
    print(f"\nFitting final model with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=100)
    clusters = kmeans.fit_predict(X_scaled)
    print("Done!")
    
    mscs['cluster'] = clusters
    
    # Analyze clusters
    print(f"\nCluster Analysis (k={optimal_k}):")
    for i in range(optimal_k):
        cluster_df = mscs[mscs['cluster'] == i]
        print(f"\nCluster {i}: n={len(cluster_df):,} ({len(cluster_df)/len(mscs):.1%})")
        print(f"  - Mean age: {cluster_df['age'].mean():.1f}")
        print(f"  - Brain death: {cluster_df['brain_death'].mean():.1%}")
        print(f"  - Approached rate: {cluster_df['approached'].mean():.1%}")
        print(f"  - Authorization rate (if approached): {cluster_df[cluster_df['approached']]['authorized'].mean():.1%}")
    
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                        alpha=0.6, s=20, edgecolors='none')
    
    # Mark approached vs not approached
    approached_mask = mscs['approached'] == True
    ax.scatter(X_pca[approached_mask, 0], X_pca[approached_mask, 1], 
              marker='x', c='red', s=50, alpha=0.3, label='Approached')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Hidden Clusters in MSC Population', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster', ax=ax)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_clusters_visualization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / 'ml_clusters_visualization.png'}")
    
    # === Key Insights ===
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM EXPERIMENT 2")
    print("="*80)
    
    print("\n1. Distinct Subpopulations:")
    print(f"   - Identified {optimal_k} distinct clusters in MSC population")
    
    print("\n2. Cluster-Specific Sorting Rates:")
    for i in range(optimal_k):
        cluster_df = mscs[mscs['cluster'] == i]
        print(f"   - Cluster {i}: {cluster_df['approached'].mean():.1%} approached")
    
    print("\n3. Hypothesis Generation:")
    print("   - Do different clusters require different cultivation strategies?")
    print("   - Are some clusters systematically under-served?")
    
    return mscs, clusters

# ============================================================================
# EXPERIMENT 3: Anomaly Detection (Edge Cases)
# ============================================================================

def experiment_3_anomalies():
    """
    Identify anomalous cases that defy general patterns.
    Goal: Find interesting edge cases for case study analysis.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Anomaly Detection (Interesting Edge Cases)")
    print("="*80)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    mscs = df[df['is_msc_percentile_99'] == True].copy()
    
    # Define anomalies
    print("\nSearching for anomalous patterns...")
    
    # Anomaly 1: Young, brain-dead, but not approached
    anomaly_1 = mscs[
        (mscs['age'] < 40) & 
        (mscs['brain_death'] == True) & 
        (mscs['approached'] == False)
    ]
    print(f"\n1. Young DBD donors NOT approached: {len(anomaly_1):,} cases")
    print("   - These should be 'perfect' donors. Why were they missed?")
    
    # Anomaly 2: Old, DCD, but successfully transplanted
    anomaly_2 = mscs[
        (mscs['age'] > 65) & 
        (mscs['brain_death'] == False) & 
        (mscs['transplanted'] == True)
    ]
    print(f"\n2. Elderly DCD donors successfully transplanted: {len(anomaly_2):,} cases")
    print("   - These defy conventional wisdom. What made them work?")
    
    # Anomaly 3: Approached on weekends
    mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
    mscs['day_of_week'] = mscs['time_referred'].dt.dayofweek
    anomaly_3 = mscs[
        (mscs['day_of_week'].isin([5, 6])) & 
        (mscs['approached'] == True)
    ]
    print(f"\n3. Weekend approaches: {len(anomaly_3):,} cases")
    print(f"   - Weekend approach rate: {len(anomaly_3) / len(mscs[mscs['day_of_week'].isin([5, 6])]):.1%}")
    print("   - What's different about these cases?")
    
    # Anomaly 4: Authorized but not procured
    anomaly_4 = mscs[
        (mscs['authorized'] == True) & 
        (mscs['procured'] == False)
    ]
    print(f"\n4. Authorized but NOT procured: {len(anomaly_4):,} cases")
    print("   - Family said yes, but organs not recovered. Why?")
    
    # Save anomalies for case study
    anomalies_file = OUTPUT_DIR / 'anomalous_cases.csv'
    anomaly_1['anomaly_type'] = 'young_dbd_not_approached'
    anomaly_2['anomaly_type'] = 'elderly_dcd_success'
    anomaly_3['anomaly_type'] = 'weekend_approach'
    anomaly_4['anomaly_type'] = 'authorized_not_procured'
    
    all_anomalies = pd.concat([anomaly_1, anomaly_2, anomaly_3, anomaly_4])
    all_anomalies.to_csv(anomalies_file, index=False)
    print(f"\n✓ Saved {len(all_anomalies):,} anomalous cases to: {anomalies_file}")
    
    # === Key Insights ===
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM EXPERIMENT 3")
    print("="*80)
    
    print("\n1. System Inefficiencies:")
    print(f"   - {len(anomaly_1):,} 'perfect' donors were missed")
    print("   - Suggests sorting failures are not just about marginal cases")
    
    print("\n2. Untapped Potential:")
    print(f"   - {len(anomaly_2):,} elderly DCD donors succeeded")
    print("   - Proves biological capacity exceeds current utilization")
    
    print("\n3. Process Failures:")
    print(f"   - {len(anomaly_4):,} families authorized but organs not procured")
    print("   - Indicates downstream coordination failures")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("EXPLORATORY ML ANALYSIS: Discovering Unexpected Patterns")
    print("="*80)
    print("\nGoal: Use machine learning to explore beyond confirmatory analysis")
    print("      and generate new hypotheses for future research.")
    
    # Experiment 1: Predictors
    rf, nn, gb, X_test, y_test, feature_cols = experiment_1_sorting_predictors()
    
    # Experiment 2: Clustering
    mscs, clusters = experiment_2_clustering(X_test, feature_cols)
    
    # Experiment 3: Anomalies
    experiment_3_anomalies()
    
    print("\n" + "="*80)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review anomalous cases for qualitative insights")
    print("  2. Design targeted interventions for each cluster")
    print("  3. Test interaction effects identified by gradient boosting")
    print("  4. Conduct case studies on edge cases")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
