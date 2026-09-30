# Two-Feature Linear Regression Model

This project extends a BMI-only linear regression model by adding `age` as a second feature.

The model is:

```text
expenses = w0 + w1 * bmi + w2 * age
```

The model is trained using:

- Normal Equation
- Gradient Descent

Both methods are evaluated using MSE, RMSE, MAE, and R², then compared with the BMI-only baseline.

## Project Structure

```text
two-feature-model/
│
├── data/
│   └── insurance.csv
├── reports/
│   └── assignment_results.csv
├── src/
│   └── two_feature_model.py
├── memo.md
├── README.md
└── requirements.txt
```

## Expected Output

The final model comparison should be similar to:

```text
Model                         w0      w1 BMI      w2 Age             MSE         RMSE        MAE      R^2
----------------------------------------------------------------------------------------------------------
BMI-only Baseline        1178.18      394.33           -    140764214.67     11864.41    9172.30   0.0394
Normal Equation         -6437.35      333.39      241.90    129359773.29     11373.64    9032.28   0.1173
Gradient Descent        -6437.35      333.39      241.90    129359773.29     11373.64    9032.28   0.1173
```
---

# Setup Instructions

## 1. Clone the Repository

### Windows

Open PowerShell, Command Prompt, or the VS Code terminal.

Navigate to the folder where you want to save the project:

```powershell
cd C:\Users\YourName\Documents
```

Clone the repository:

```powershell
git clone https://github.com/hngo13-cell/two-feature-model.git
```

Move into the project folder:

```powershell
cd two-feature-model
```

### macOS

Open Terminal.

Navigate to the folder where you want to save the project:

```bash
cd ~/Documents
```

Clone the repository:

```bash
git clone https://github.com/hngo13-cell/two-feature-model.git
```

Move into the project folder:

```bash
cd two-feature-model
```

---

## 2. Open the Project in Visual Studio Code

Run:

```bash
code .
```

If the `code` command is not available, open Visual Studio Code manually and select:

```text
File → Open Folder → two-feature-model
```

---

## 3. Create a Virtual Environment

### Windows

Run:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy Bypass -Scope Process
```

Then activate again:

```powershell
.\venv\Scripts\activate
```

After activation, the terminal should begin with:

```text
(venv)
```

### macOS

Run:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

After activation, the terminal should begin with:

```text
(venv)
```

---

## 4. Install the Required Packages

With the virtual environment activated, run:

### Windows

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## 5. Run the Model

Make sure you are in the project root directory:

```text
two-feature-model
```

### Windows

Run:

```powershell
python .\src\two_feature_model.py
```

### macOS

Run:

```bash
python3 src/two_feature_model.py
```

The script will:

1. Load `bmi`, `age`, and `expenses`.
2. Train the two-feature model using the Normal Equation.
3. Train the same model using Gradient Descent.
4. Calculate MSE, RMSE, MAE, and R².
5. Compare the models with the BMI-only baseline.
6. Export the results to:

```text
reports/assignment_results.csv
```

---

## Expected Output

The final output should be similar to:

```text
Model                         w0      w1 BMI      w2 Age        R²
-----------------------------------------------------------------
BMI-only Baseline        1178.18      394.33           -     0.0394
Normal Equation         -6437.35      333.39      241.90     0.1173
Gradient Descent        -6437.35      333.39      241.90     0.1173
```

---

## Output Files

The comparison results are saved to:

```text
reports/assignment_results.csv
```

The written interpretation is included in:

```text
memo.md
```
