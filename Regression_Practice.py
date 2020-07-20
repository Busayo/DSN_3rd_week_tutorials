# DSN Practice for Lesson 11, R_squared line.
# Writing the regression algorithm from scratch
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

xes = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
yes = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(xes, yes):
	m = (((mean(xes) * mean(yes)) - mean(xes * yes)) /
	     ((mean(xes) ** 2) - mean(xes ** 2)))
	b = mean(yes) - m * mean(xes)
	return m, b


def squared_error(yes_orig, yes_line):
	return sum((yes_line - yes_orig) ** 2)


def coefficient_of_determination(yes_orig, yes_line):
	y_mean_line = [mean(yes_orig) for y in yes_orig]
	squared_error_regr = squared_error(yes_orig, yes_line)
	squared_error_y_mean = squared_error(yes_orig, y_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)


m, b = best_fit_slope_and_intercept(xes, yes)

regression_line = [(m * x) + b for x in xes]

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(yes, regression_line)
print(r_squared)

plt.scatter(xes, yes)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xes, regression_line)
plt.show()
