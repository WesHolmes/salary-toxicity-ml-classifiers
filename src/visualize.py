import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from salary_predictor import SalaryPredictor

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get the project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent

def load_data():
    """Load and split the salary data."""
    salary_data = pd.read_csv(PROJECT_ROOT / 'dat' / 'salary.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        salary_data.drop(axis=1, columns=['class']), 
        salary_data['class'], 
        test_size=0.3, 
        random_state=42
    )
    return salary_data, X_train, X_test, y_train, y_test

def visualize_class_distribution(data: pd.DataFrame):
    """Visualize the distribution of salary classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=data, x='class', ax=axes[0])
    axes[0].set_title('Distribution of Salary Classes', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class (0: <=50k, 1: >50k)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticklabels(['<=50k', '>50k'])
    
    # Pie chart
    class_counts = data['class'].value_counts()
    axes[1].pie(class_counts, labels=['<=50k', '>50k'], autopct='%1.1f%%', 
                startangle=90, colors=['#ff9999', '#66b3ff'])
    axes[1].set_title('Salary Class Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    plt.close()

def visualize_numerical_features(data: pd.DataFrame):
    """Visualize distributions of numerical features."""
    numerical_cols = ['age', 'education_years', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if col in data.columns:
            sns.histplot(data=data, x=col, hue='class', bins=30, ax=axes[idx], alpha=0.7)
            axes[idx].set_title(f'Distribution of {col} by Salary Class', fontweight='bold')
            axes[idx].set_xlabel(col.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend(title='Salary', labels=['<=50k', '>50k'])
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'numerical_features.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: numerical_features.png")
    plt.close()

def visualize_categorical_features(data: pd.DataFrame):
    """Visualize distributions of categorical features."""
    categorical_cols = ['work_class', 'education', 'marital', 'race', 'sex']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        if col in data.columns:
            # Count plot with salary class as hue
            sns.countplot(data=data, x=col, hue='class', ax=axes[idx])
            axes[idx].set_title(f'{col.replace("_", " ").title()} by Salary Class', fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].legend(title='Salary', labels=['<=50k', '>50k'])
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'categorical_features.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: categorical_features.png")
    plt.close()

def visualize_correlation(data: pd.DataFrame):
    """Visualize correlation matrix of numerical features."""
    numerical_cols = ['age', 'education_years', 'capital_gain', 'capital_loss', 'hours_per_week', 'class']
    numerical_data = data[numerical_cols]
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()

def visualize_confusion_matrix(y_test, y_pred):
    """Visualize the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50k', '>50k'], 
                yticklabels=['<=50k', '>50k'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.close()

def visualize_roc_curve(y_test, y_pred_proba):
    """Visualize the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curve.png")
    plt.close()

def visualize_feature_importance(predictor: SalaryPredictor, feature_names: list):
    """Visualize feature importance from logistic regression coefficients."""
    # Get coefficients from the logistic regression model
    coefficients = predictor.lrbc.coef_[0]
    
    # Get feature names (this is a simplified version - you may need to adjust based on your preprocessing)
    # For one-hot encoded features, you'd need to get the feature names from the encoder
    n_features = len(coefficients)
    
    # Create indices for features
    indices = np.argsort(np.abs(coefficients))[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), coefficients[indices])
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('Top 20 Feature Importances (Logistic Regression Coefficients)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Generating Data Visualizations")
    print("=" * 60)
    
    # Load data
    print("\n[1/7] Loading data...")
    data, X_train, X_test, y_train, y_test = load_data()
    print(f"   Loaded {len(data)} total records")
    print(f"   Training set: {len(X_train)} records")
    print(f"   Test set: {len(X_test)} records")
    
    # Data exploration visualizations
    print("\n[2/7] Visualizing class distribution...")
    visualize_class_distribution(data)
    
    print("\n[3/7] Visualizing numerical features...")
    visualize_numerical_features(data)
    
    print("\n[4/7] Visualizing categorical features...")
    visualize_categorical_features(data)
    
    print("\n[5/7] Visualizing correlation matrix...")
    visualize_correlation(data)
    
    # Model performance visualizations
    print("\n[6/7] Training model and generating performance visualizations...")
    predictor = SalaryPredictor(X_train, y_train)
    y_pred = predictor.classify(X_test)
    
    # Confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
    
    # ROC curve (need probability predictions)
    y_pred_proba = predictor.lrbc.predict_proba(predictor.preprocess(X_test, False))[:, 1]
    visualize_roc_curve(y_test, y_pred_proba)
    
    print("\n[7/7] Visualizing feature importance...")
    # Note: Feature importance visualization is simplified - you may want to enhance it
    # to show actual feature names from your preprocessing pipeline
    visualize_feature_importance(predictor, [])
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - class_distribution.png")
    print("  - numerical_features.png")
    print("  - categorical_features.png")
    print("  - correlation_matrix.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")

if __name__ == '__main__':
    main()

