from __future__ import annotations

import csv
import math
from pathlib import Path


# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "insurance.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "assignment_results.csv"

LEARNING_RATE = 0.05
EPOCHS = 10000
LOG_EVERY = 1000


# 1. Load bmi, age, expenses
def load_data(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    bmi: list[float] = []
    age: list[float] = []
    expenses: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bmi.append(float(row["bmi"]))
            age.append(float(row["age"]))
            expenses.append(float(row["expenses"]))

    return bmi, age, expenses


# General linear solver
def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(A)
    aug = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix in normal equation.")

        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


# 2. Two-feature Normal Equation
def fit_normal_equation_two_features(
    bmi: list[float],
    age: list[float],
    y: list[float],
) -> tuple[float, float, float]:

    X = [[1.0, b, a] for b, a in zip(bmi, age)]
    rows = len(X)
    cols = len(X[0])

    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]

    for i in range(cols):
        for j in range(cols):
            xtx[i][j] = sum(X[r][i] * X[r][j] for r in range(rows))
        xty[i] = sum(X[r][i] * y[r] for r in range(rows))

    w0, w1, w2 = solve_linear_system(xtx, xty)
    return w0, w1, w2


# Standardize feature
def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5

    if std == 0:
        return [0.0 for _ in values], mean, 1.0

    return [(v - mean) / std for v in values], mean, std


# 3. Two-feature Gradient Descent
def fit_gradient_descent_two_features(
    bmi_raw: list[float],
    age_raw: list[float],
    y: list[float],
    learning_rate: float,
    epochs: int,
) -> tuple[float, float, float]:

    bmi, bmi_mean, bmi_std = standardize(bmi_raw)
    age, age_mean, age_std = standardize(age_raw)

    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    n = len(y)

    for epoch in range(1, epochs + 1):
        preds = [w0 + w1 * b + w2 * a for b, a in zip(bmi, age)]
        errors = [p - yi for p, yi in zip(preds, y)]

        grad_w0 = (2.0 / n) * sum(errors)
        grad_w1 = (2.0 / n) * sum(err * b for err, b in zip(errors, bmi))
        grad_w2 = (2.0 / n) * sum(err * a for err, a in zip(errors, age))

        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        if epoch % LOG_EVERY == 0 or epoch == 1:
            current_preds = [w0 + w1 * b + w2 * a for b, a in zip(bmi, age)]
            current_mse = sum((yi - pi) ** 2 for yi, pi in zip(y, current_preds)) / n
            print(f"epoch={epoch:5d} mse={current_mse:.2f}")

    # Convert to original units
    w1_original = w1 / bmi_std
    w2_original = w2 / age_std
    w0_original = w0 - (w1 * bmi_mean / bmi_std) - (w2 * age_mean / age_std)

    return w0_original, w1_original, w2_original


# 4. Evaluation metrics
def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


# 5. BMI-only baseline
def fit_normal_equation_single_feature(
    x: list[float],
    y: list[float],
) -> tuple[float, float]:

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    a = float(n)
    b = sum_x
    c = sum_x
    d = sum_x2

    det = a * d - b * c
    if det == 0:
        raise ValueError("Singular matrix in normal equation.")

    inv_xtx = [[d / det, -b / det], [-c / det, a / det]]
    xty = [sum_y, sum_xy]

    w0 = inv_xtx[0][0] * xty[0] + inv_xtx[0][1] * xty[1]
    w1 = inv_xtx[1][0] * xty[0] + inv_xtx[1][1] * xty[1]

    return w0, w1


# Predictions
def predict_two_features(
    bmi: list[float],
    age: list[float],
    w0: float,
    w1: float,
    w2: float,
) -> list[float]:
    return [w0 + w1 * b + w2 * a for b, a in zip(bmi, age)]


def predict_single_feature(
    x: list[float],
    w0: float,
    w1: float,
) -> list[float]:
    return [w0 + w1 * xi for xi in x]


# Model metrics
def evaluate_model(y: list[float], preds: list[float]) -> dict[str, float]:
    return {
        "MSE": mse(y, preds),
        "RMSE": rmse(y, preds),
        "MAE": mae(y, preds),
        "R2": r2_score(y, preds),
    }


# 6. Compare and export
def main() -> None:
    bmi, age, expenses = load_data(CSV_PATH)

    print(f"Dataset: {CSV_PATH}")
    print(f"Rows: {len(expenses)}")
    print()

    # Normal Equation
    w0_ne, w1_ne, w2_ne = fit_normal_equation_two_features(bmi, age, expenses)
    preds_ne = predict_two_features(bmi, age, w0_ne, w1_ne, w2_ne)
    metrics_ne = evaluate_model(expenses, preds_ne)

    # Gradient Descent
    print("Gradient Descent training:")
    w0_gd, w1_gd, w2_gd = fit_gradient_descent_two_features(
        bmi, age, expenses, LEARNING_RATE, EPOCHS
    )
    preds_gd = predict_two_features(bmi, age, w0_gd, w1_gd, w2_gd)
    metrics_gd = evaluate_model(expenses, preds_gd)

    # BMI-only baseline
    w0_base, w1_base = fit_normal_equation_single_feature(bmi, expenses)
    preds_base = predict_single_feature(bmi, w0_base, w1_base)
    metrics_base = evaluate_model(expenses, preds_base)

    results = [
        ["BMI-only Baseline", w0_base, w1_base, "", metrics_base["MSE"],
         metrics_base["RMSE"], metrics_base["MAE"], metrics_base["R2"]],

        ["Normal Equation", w0_ne, w1_ne, w2_ne, metrics_ne["MSE"],
         metrics_ne["RMSE"], metrics_ne["MAE"], metrics_ne["R2"]],

        ["Gradient Descent", w0_gd, w1_gd, w2_gd, metrics_gd["MSE"],
         metrics_gd["RMSE"], metrics_gd["MAE"], metrics_gd["R2"]],
    ]

    print()
    print("Model comparison")
    print(
        f"{'Model':<22}{'w0':>12}{'w1 BMI':>12}{'w2 Age':>12}"
        f"{'MSE':>15}{'RMSE':>12}{'MAE':>12}{'R^2':>10}"
    )
    print("-" * 107)

    for row in results:
        w2_text = "-" if row[3] == "" else f"{row[3]:.2f}"
        print(
            f"{row[0]:<22}{row[1]:>12.2f}{row[2]:>12.2f}{w2_text:>12}"
            f"{row[4]:>15.2f}{row[5]:>12.2f}{row[6]:>12.2f}{row[7]:>10.4f}"
        )

    # Export CSV
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "w0", "w1_bmi", "w2_age",
            "MSE", "RMSE", "MAE", "R2"
        ])
        writer.writerows(results)

    print()
    print(f"Results saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()