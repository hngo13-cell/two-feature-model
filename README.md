# BMI and Age Expense Regression

This project extends a single-feature linear regression model into a two-feature model for predicting medical expenses using **BMI** and **age**.

The original baseline model uses only BMI:

```text
expenses = w0 + w1 * bmi
```

The extended model uses both BMI and age:

```text
expenses = w0 + w1 * bmi + w2 * age
```

The two-feature model is trained using two different methods:

- Normal Equation
- Gradient Descent

The performance of both methods is then compared with the original BMI-only baseline.

---

## Project Objectives

The main goals of this project are to:

1. Load BMI, age, and medical expense data from `insurance.csv`.
2. Extend the BMI-only regression model by adding age as a second feature.
3. Fit the two-feature model using the Normal Equation.
4. Fit the same model using Gradient Descent.
5. Evaluate the models using MSE, RMSE, MAE, and R².
6. Compare the two-feature models with the BMI-only baseline.
7. Export the final comparison results to a CSV file.
8. Interpret the results in a short written memo.

---

## Project Structure

```text
bmi-age-expense-regression/
│
├── data/
│   └── insurance.csv
│
├── reports/
│   └── assignment_results.csv
│
├── src/
│   └── two_feature_model.py
│
├── memo.md
├── README.md
└── requirements.txt
```

### File Descriptions

- `data/insurance.csv`  
  Contains the insurance dataset used for model training and evaluation.

- `src/two_feature_model.py`  
  Contains the complete implementation of the BMI-only baseline, two-feature Normal Equation model, Gradient Descent model, evaluation metrics, comparison table, and CSV export.

- `reports/assignment_results.csv`  
  Contains the model coefficients and evaluation metrics for all three models.

- `memo.md`  
  Contains the written interpretation of the model results.

- `requirements.txt`  
  Contains the Python package requirements for the project.

---

## Dataset

The project uses three variables from the insurance dataset:

| Variable | Description |
|---|---|
| `bmi` | Body Mass Index |
| `age` | Age of the individual |
| `expenses` | Medical expenses used as the target variable |

The target variable is:

```text
expenses
```

The two input features are:

```text
bmi
age
```

---

## Models

### 1. BMI-Only Baseline

The baseline model uses BMI as the only input feature:

```text
expenses = w0 + w1 * bmi
```

This model provides a reference point for determining whether adding age improves model performance.

---

### 2. Two-Feature Normal Equation

The extended regression model is:

```text
expenses = w0 + w1 * bmi + w2 * age
```

For the Normal Equation, the design matrix contains an intercept column together with BMI and age:

```text
[1, bmi, age]
```

The general Normal Equation is:

```text
theta = (X^T X)^-1 X^T y
```

Because the model contains two features and an intercept, the implementation uses a general linear system solver rather than the 2x2 shortcut used for the single-feature model.

The resulting coefficients are:

- `w0`: intercept
- `w1`: BMI coefficient
- `w2`: age coefficient

---

### 3. Two-Feature Gradient Descent

Gradient Descent fits the same model:

```text
expenses = w0 + w1 * bmi + w2 * age
```

Unlike the Normal Equation, Gradient Descent finds the coefficients iteratively.

Before training, both BMI and age are standardized:

```text
standardized value = (value - mean) / standard deviation
```

The model begins with initial weights of zero and repeatedly:

1. Makes predictions.
2. Calculates prediction errors.
3. Calculates the gradients.
4. Updates the coefficients.
5. Repeats the process over multiple epochs.

The project uses:

```text
Learning Rate = 0.05
Epochs = 10000
```

After training, the Gradient Descent coefficients are converted back to the original BMI and age units so they can be directly compared with the Normal Equation coefficients.

---

## Evaluation Metrics

Each model is evaluated using four regression metrics.

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted expenses.

```text
Lower MSE = smaller prediction errors
```

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

```text
Lower RMSE = better prediction accuracy
```

Because RMSE is in the same unit as the target variable, it is easier to interpret than MSE.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted expenses.

```text
Lower MAE = smaller average prediction error
```

### R-Squared (R²)

R² measures how much of the variation in medical expenses is explained by the model.

```text
Higher R² = more variation explained by the model
```

---

## Results

The models produced the following coefficients and R² values:

| Model | w0 | w1 (BMI) | w2 (Age) | R² |
|---|---:|---:|---:|---:|
| BMI-only Baseline | 1178.18 | 394.33 | — | 0.0394 |
| Normal Equation | -6437.35 | 333.39 | 241.90 | 0.1173 |
| Gradient Descent | -6437.35 | 333.39 | 241.90 | 0.1173 |

The BMI-only model explains approximately:

```text
3.94%
```

of the variation in medical expenses.

After adding age, the two-feature model explains approximately:

```text
11.73%
```

of the variation.

The improvement in R² is:

```text
0.1173 - 0.0394 = 0.0779
```

This represents an improvement of approximately **7.79 percentage points in explained variation**.

---

## Normal Equation vs. Gradient Descent

The Normal Equation and Gradient Descent produced almost identical coefficients:

```text
Normal Equation
w0 ≈ -6437.35
w1 ≈ 333.39
w2 ≈ 241.90

Gradient Descent
w0 ≈ -6437.35
w1 ≈ 333.39
w2 ≈ 241.90
```

This result is expected because both methods are solving the same Ordinary Least Squares regression problem.

The main difference is how they find the coefficients:

```text
Normal Equation
      ↓
Direct analytical solution
      ↓
Final coefficients
```

while Gradient Descent uses:

```text
Initial coefficients
      ↓
Calculate predictions
      ↓
Calculate errors
      ↓
Update coefficients
      ↓
Repeat
      ↓
Final coefficients
```

The Normal Equation finds the solution directly, while Gradient Descent approaches the solution through repeated updates.

---

## Coefficient Interpretation

The age coefficient is approximately:

```text
w2 = 241.90
```

This means that, **holding BMI constant**, each additional year of age is associated with approximately **$241.90 higher predicted medical expenses**.

The BMI coefficient in the two-feature model is approximately:

```text
w1 = 333.39
```

This means that, **holding age constant**, a one-unit increase in BMI is associated with approximately **$333.39 higher predicted medical expenses**.

These coefficients describe relationships in the regression model and should not automatically be interpreted as causal effects.

---

## Model Performance

Adding age improves the regression model compared with using BMI alone.

However, the two-feature model has an R² of approximately:

```text
0.1173
```

This means that BMI and age together explain only about **11.73% of the variation in medical expenses**.

Most of the variation in medical expenses is therefore still not explained by this model.

The model is useful as a simple regression baseline and demonstrates the effect of adding another feature, but additional relevant variables would be needed before considering the model strong enough for real-world prediction.

---

## Setup Instructions

### 1. Clone the Repository

Open a terminal and navigate to the folder where you want to save the project.

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd bmi-age-expense-regression
```

---

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

After activation, the terminal should begin with:

```text
(venv)
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Requirements

With the virtual environment activated, run:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Model

From the project root directory, run:

```bash
python src/two_feature_model.py
```

On Windows PowerShell, the command can also be written as:

```powershell
python .\src\two_feature_model.py
```

The script will:

1. Load BMI, age, and expenses.
2. Train the two-feature Normal Equation model.
3. Train the two-feature Gradient Descent model.
4. Calculate MSE, RMSE, MAE, and R².
5. Train and evaluate the BMI-only baseline.
6. Print a comparison table in the terminal.
7. Export the results to a CSV file.

---

## Example Output

The final terminal output includes a comparison similar to:

```text
Model                         w0      w1 BMI      w2 Age            MSE        RMSE         MAE       R²
-----------------------------------------------------------------------------------------------------------
BMI-only Baseline        1178.18      394.33           -          ...          ...         ...    0.0394
Normal Equation         -6437.35      333.39      241.90          ...          ...         ...    0.1173
Gradient Descent        -6437.35      333.39      241.90          ...          ...         ...    0.1173
```

During Gradient Descent training, the script also displays MSE checkpoints by epoch to show the convergence process.

---

## Exported Results

After the script finishes running, the model comparison is automatically written to:

```text
reports/assignment_results.csv
```

The CSV contains:

```text
Model
w0
w1_bmi
w2_age
MSE
RMSE
MAE
R2
```

The file can be opened in Excel, Google Sheets, or another spreadsheet program for easier viewing.

---

## Written Memo

A short written interpretation of the results is included in:

```text
memo.md
```

The memo discusses:

- Whether adding age improved R².
- The difference between the BMI-only and two-feature models.
- Why the Normal Equation and Gradient Descent produce similar coefficients.
- The interpretation of the age coefficient.
- Whether the two-feature model is strong enough for deployment.

---

## Summary

This project demonstrates how a single-feature linear regression model can be extended by adding a second feature.

The original BMI-only model had an R² of approximately **0.0394**, while the BMI-and-age model increased R² to approximately **0.1173**.

Both the Normal Equation and Gradient Descent produced almost identical coefficients, showing that the two approaches can reach the same Ordinary Least Squares solution through different training methods.

Although adding age improved the model, BMI and age alone still explain only a relatively small portion of the variation in medical expenses. The project therefore demonstrates both the value of adding relevant features and the limitations of a simple two-feature regression model.