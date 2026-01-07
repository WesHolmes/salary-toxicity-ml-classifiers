# ML Classifiers - Binary Classification Project

This machine learning project implements two production ready binary classification systems: a **Salary Predictor** using Logistic Regression and a **Toxicity Filter** using Naive Bayes Classification. The project demonstrates end-to-end ML workflows including data preprocessing, model training, evaluation, and comprehensive visualization.

## Features

- **Dual classifier system**: Logistic Regression for salary prediction and Multinomial Naive Bayes for toxicity detection

- **Robust preprocessing**: Feature engineering with one-hot encoding, standardization, and text vectorization

- **Comprehensive testing**: Unit tests with performance benchmarks and validation

- **Data visualization**: Automated EDA and model performance visualizations

- **Production-ready code**: Type hints, error handling, and clean architecture

- **Reproducible results**: Fixed random seeds for consistent train/test splits

- **Type safety**: Strict mypy configuration for type checking

## Project Structure

```
ml-classifiers/
├── dat/                             # Data directory
│   ├── salary.csv                  # Adult Census Income dataset (~48k samples)
│   └── wiki_talk.csv               # Wikipedia talk page comments (~560k samples)
├── src/                             # Source code
│   ├── salary_predictor.py         # Salary prediction classifier
│   ├── toxicity_filter.py          # Toxicity detection classifier
│   ├── visualize.py                # Data visualization utilities
│   ├── classifier_tests.py         # Unit tests and benchmarks
│   ├── mypy.ini                    # Type checking configuration
│   └── pytest.ini                  # Testing configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

Generated visualizations (saved to project root):
- `class_distribution.png`
- `numerical_features.png`
- `categorical_features.png`
- `correlation_matrix.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone and Install

```bash
git clone <repository-url>
cd ml-classifiers
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
cd src
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies installed successfully')"
```

### 3. Run Tests

```bash
cd src
python classifier_tests.py
```

Or using pytest:

```bash
cd src
pytest classifier_tests.py -v
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation and analysis |
| numpy | ≥1.21.0 | Numerical computations |
| scikit-learn | ≥1.0.0 | Machine learning algorithms |
| matplotlib | ≥3.5.0 | Plotting and visualization |
| seaborn | ≥0.12.0 | Statistical data visualization |

## Usage

### Quick Start - Salary Predictor

```python
from src.salary_predictor import SalaryPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and split data
data = pd.read_csv('dat/salary.csv')
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
predictor = SalaryPredictor(X_train, y_train)

# Make predictions
predictions = predictor.classify(X_test)

# Evaluate model
report_str, report_dict = predictor.test_model(X_test, y_test)
print(report_str)
```

### Quick Start - Toxicity Filter

```python
from src.toxicity_filter import ToxicityFilter
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and split data
data = pd.read_csv('dat/wiki_talk.csv')
X = data['comment']
y = data['toxic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337)

# Train model
filter_model = ToxicityFilter(X_train, y_train)

# Classify comments
test_comments = [
    "This is a helpful and constructive comment.",
    "You're an idiot and this article is terrible."
]
predictions = filter_model.classify(test_comments)
# Returns: [0, 1] (non-toxic, toxic)

# Evaluate model
report_str, report_dict = filter_model.test_model(X_test, y_test)
print(report_str)
```

### Generate Visualizations

```bash
cd src
python visualize.py
```

All visualizations are automatically saved to the project root as high-resolution PNG files (300 DPI).

### Making Predictions on New Data

**Salary Predictor:**

```python
# New individual data
new_data = pd.DataFrame({
    'age': [45],
    'work_class': ['Private'],
    'education': ['Masters'],
    'education_years': [14],
    'marital': ['Married-civ-spouse'],
    'occupation_code': ['Exec-managerial'],
    'relationship': ['Husband'],
    'race': ['White'],
    'sex': ['Male'],
    'capital_gain': [0],
    'capital_loss': [0],
    'hours_per_week': [50],
    'country': ['United-States']
})

prediction = predictor.classify(new_data)
# Returns: [0] or [1] (≤50k or >50k)
```

**Toxicity Filter:**

```python
new_comments = ["This is a great article, thanks for sharing!"]
predictions = filter_model.classify(new_comments)
# Returns: [0] (non-toxic)
```

## Component Details

### Salary Predictor

Binary classification model predicting annual salary above/below $50,000 using Logistic Regression.

**Algorithm:**
- Model: Logistic Regression (sklearn)
- Optimization: Maximum likelihood estimation with iterative solver
- Max Iterations: 5000 (configurable in `__init__`)

**Features Processed:**

| Feature Type | Examples |
|--------------|----------|
| **Categorical** | work_class, education, marital, relationship, race, sex, country, occupation_code |
| **Numerical** | age, education_years, capital_gain, capital_loss, hours_per_week |

**Preprocessing Pipeline:**

1. **Data Cleaning**: Strips whitespace from string columns, preserves original DataFrame
2. **Categorical Encoding**: One-Hot Encoding with `handle_unknown='ignore'` for inference
3. **Numerical Standardization**: StandardScaler for zero-mean, unit-variance normalization
4. **Feature Combination**: Horizontal stacking of standardized numerical and one-hot encoded categorical features

**Methods:**

- `__init__(X_train, y_train)`: Trains the logistic regression model on preprocessed training data
- `preprocess(features, training=False)`: Transforms raw features into numerical format
- `classify(X_test)`: Returns binary predictions (0: ≤50k, 1: >50k)
- `test_model(X_test, y_test)`: Returns classification report in string and dictionary formats

### Toxicity Filter

Binary text classification model detecting toxic/offensive comments using Multinomial Naive Bayes.

**Algorithm:**
- Model: Multinomial Naive Bayes (sklearn)
- Text Representation: Bag-of-Words (CountVectorizer)
- Stop Words: English stop words removed

**Preprocessing Pipeline:**

1. **Text Vectorization**: `CountVectorizer` creates vocabulary from training text, removes stop words
2. **Training**: MultinomialNB learns word frequency distributions for toxic vs. non-toxic classes

**Methods:**

- `__init__(text_train, labels_train)`: Trains the Naive Bayes classifier on vectorized training text
- `classify(text_test)`: Returns binary predictions (0: non-toxic, 1: toxic)
- `test_model(text_test, labels_test)`: Returns classification report in string and dictionary formats

**Attributes:**

- `vectorizer`: sklearn `CountVectorizer` instance with learned vocabulary in `vectorizer.vocabulary_` (>100k words typically)

### Visualization Module

The `visualize.py` script generates comprehensive visualizations:

1. **Class Distribution**: Bar chart and pie chart showing target class distribution
2. **Numerical Features**: Histogram distributions color-coded by target class
3. **Categorical Features**: Count plots stacked by target class
4. **Correlation Matrix**: Heatmap of pairwise correlations between numerical features
5. **Confusion Matrix**: Model performance showing true vs. predicted labels
6. **ROC Curve**: Receiver Operating Characteristic curve with AUC score
7. **Feature Importance**: Top 20 logistic regression coefficients

## Data Schema

### Salary Dataset (`dat/salary.csv`)

Adult Census Income dataset from 1994 U.S. Census Bureau database.

| Column | Type | Description |
|--------|------|-------------|
| age | INTEGER | Age of individual (17-90) |
| work_class | TEXT | Employment type (Private, State-gov, Self-emp-not-inc, etc.) |
| education | TEXT | Education level (Bachelors, HS-grad, Masters, etc.) |
| education_years | INTEGER | Years of education (1-16) |
| marital | TEXT | Marital status (Married-civ-spouse, Divorced, Never-married, etc.) |
| occupation_code | TEXT | Occupation category code |
| relationship | TEXT | Relationship status (Husband, Wife, Not-in-family, etc.) |
| race | TEXT | Race category |
| sex | TEXT | Sex (Male, Female) |
| capital_gain | INTEGER | Capital gains (typically 0 or >0) |
| capital_loss | INTEGER | Capital losses (typically 0 or >0) |
| hours_per_week | INTEGER | Hours worked per week (1-99) |
| country | TEXT | Country of origin |
| class | INTEGER | Target variable (0: ≤50k, 1: >50k) |

**Dataset Size**: ~48,000 samples

### Wiki Talk Dataset (`dat/wiki_talk.csv`)

Wikipedia talk page comments dataset for toxicity detection.

| Column | Type | Description |
|--------|------|-------------|
| comment | TEXT | Forum comment text |
| toxic | INTEGER | Target variable (0: non-toxic, 1: toxic) |

**Dataset Size**: ~560,000+ samples

## Model Performance

### Salary Predictor

- **Test Accuracy**: >84% (validated in performance tests)
- **Model Type**: Logistic Regression
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score (per class)

### Toxicity Filter

- **Test Accuracy**: >92% (validated in performance tests)
- **Model Type**: Multinomial Naive Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score (per class)

*Note: Actual performance may vary based on data distribution and hyperparameters. Run the test suite for current performance metrics.*

## Test Configuration

- **Train/Test Split**: 70/30 for both datasets
- **Random Seeds**: 
  - Toxicity Filter: 1337
  - Salary Predictor: 42
- **Verbose Mode**: Set `VERBOSE = True` in `classifier_tests.py` for detailed output

**Test Coverage:**

- `test_toxicity_filter_vectorizer`: Validates CountVectorizer initialization and vocabulary size
- `test_toxicity_filter_classification`: Tests classification on sample comments
- `test_toxicity_filter_performance`: Validates test accuracy >92%
- `test_salary_preprocessing`: Ensures preprocessing doesn't modify original DataFrames
- `test_salary_performance_easy`: Validates test accuracy >80%
- `test_salary_performance_med`: Validates test accuracy >82%
- `test_salary_performance_hard`: Validates test accuracy >84%

## Technical Details

### Preprocessing Philosophy

**Training vs. Inference:**
- All preprocessing transformers (scalers, encoders) are fit **only** on training data
- During inference, transformers use learned statistics/parameters from training
- Prevents data leakage and ensures realistic performance evaluation

**Data Integrity:**
- Original DataFrames are never modified in-place
- All preprocessing operations work on copies to preserve input data

### Model Selection Rationale

**Logistic Regression for Salary Prediction:**
- Interpretable coefficients
- Works well with mixed categorical/numerical features
- Fast training and inference
- Good baseline for binary classification

**Naive Bayes for Text Classification:**
- Excellent for high-dimensional sparse text data
- Fast training and prediction
- Handles large vocabularies efficiently
- Well-suited for bag-of-words representations

### Type Safety

The codebase uses type hints throughout with strict mypy configuration:
- All function parameters and return types are annotated
- Type checking enforced via `mypy.ini` configuration
- `numpy.typing` used for array type annotations

## Development

### Type Checking

```bash
cd src
mypy salary_predictor.py toxicity_filter.py
```

### Code Style

The project follows Python PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` or `pylint` for linting

### Extending the Project

**Adding New Features:**

1. Modify the `preprocess()` method to include new feature engineering steps
2. Ensure feature names are consistent between training and inference

Example:

```python
def preprocess(self, features: pd.DataFrame, training: bool = False) -> npt.NDArray:
    # Add new feature engineering
    features['new_feature'] = features['col1'] * features['col2']
    # Continue with existing preprocessing...
```

**Hyperparameter Tuning:**

- Modify model initialization in `__init__` methods
- Consider using `GridSearchCV` or `RandomizedSearchCV` from sklearn

```python
# In SalaryPredictor.__init__
self.lrbc = LogisticRegression(max_iter=5000, C=0.1, penalty='l2')
```

**Adding Visualizations:**

1. Add new functions to `visualize.py` following existing patterns
2. Call new functions in the `main()` function

```python
def visualize_new_plot(data: pd.DataFrame):
    # Your visualization code
    plt.savefig(PROJECT_ROOT / 'new_plot.png')

# In main()
visualize_new_plot(data)
```

## Troubleshooting

### Common Issues

1. **"No module named 'sklearn'" error**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version is 3.8 or higher

2. **"FileNotFoundError: dat/salary.csv"**
   - Ensure you're running scripts from the `src/` directory or adjust paths
   - Verify data files exist in the `dat/` directory

3. **"ValueError: Input contains NaN"**
   - Check for missing values in your data
   - Ensure preprocessing handles missing values appropriately

4. **"MemoryError" when loading data**
   - The wiki_talk dataset is large (~560k samples)
   - Consider using a subset for development: `data = data.sample(n=10000)`

5. **Low model accuracy**
   - Verify train/test split is correct (70/30)
   - Check that random seeds match test configuration
   - Ensure data preprocessing is working correctly

6. **Type checking errors with mypy**
   - Run `mypy` with `--ignore-missing-imports` if third-party types are missing
   - Verify `mypy.ini` configuration is correct

7. **Visualization files not generating**
   - Ensure `matplotlib` backend is available
   - Check write permissions in project root directory
   - Verify `PROJECT_ROOT` path is correctly resolved

### Debug Mode

Enable detailed logging in tests:

```python
# In classifier_tests.py
VERBOSE = True  # Set to True for detailed test output
```

Enable debug logging in visualization:

```python
# In visualize.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues or questions:

1. **Check Documentation**: Review this README and code comments
2. **Run Tests**: Execute `python classifier_tests.py` to verify setup
3. **Review Logs**: Check console output for specific error messages
4. **Data Verification**: Ensure data files are correctly formatted and accessible

## Contributors

- **Westley Holmes**
- **Nick**
- **Matt**

## License

[Specify your license here]

## Acknowledgments

- **Adult Census Income Dataset**: UCI Machine Learning Repository
- **Wikipedia Talk Pages Dataset**: Wikimedia Foundation
