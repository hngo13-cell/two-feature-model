# Two-Feature Linear Regression Memo

Adding age improved the model compared with the BMI-only baseline. The BMI-only model had an R² of 0.0394, while the BMI-and-age model had an R² of 0.1173. This is an improvement of about 0.0779, or 7.79 percentage points in explained variation. This means that using age together with BMI explains more of the differences in medical expenses than using BMI alone.

The Normal Equation and Gradient Descent produced almost identical model weights. The Normal Equation produced approximately w0 = -6437.35, w1 = 333.39 for BMI, and w2 = 241.90 for age. Gradient Descent produced nearly the same values. This is expected because both methods are solving the same Ordinary Least Squares problem and are trying to minimize the same squared prediction error, although they use different methods to find the coefficients.

The age coefficient is positive and is approximately 241.90. In plain language, holding BMI constant, each additional year of age is associated with about $241.90 higher predicted medical expenses. However, this relationship should not be interpreted as proof that age directly causes the increase in expenses.

Although adding age improved the model, the R² is still only about 0.1173. This means that the model explains only about 11.73% of the variation in medical expenses. Therefore, I would not consider this two-feature model strong enough to deploy by itself, and additional useful features should be considered before using it for real-world predictions.