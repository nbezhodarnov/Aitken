import numpy as np
import matplotlib.pyplot as plt
import math
	
class Aitken():
	x = np.array([], dtype = float)
	y = np.array([], dtype = float)
	n = 0
	L = None
	
	def __init__(self, x_array_input, y_array_input, count):
		self.x = x_array_input
		self.y = y_array_input
		self.n = count
		self.L = np.zeros((count, count), dtype = float)
		for i in range(count):
			self.L[i][0] = self.y[i]
		
	def Aitken_calculate(self, x_input):
		for i in range(1, self.n):
			for j in range(i, self.n):
				self.L[j][i] = (self.L[j - 1][i - 1] * (self.x[j] - x_input) - self.L[j][i - 1] * (self.x[j - i] - x_input)) / (self.x[j] - self.x[j - i])
		return self.L[self.n - 1][self.n - 1]
		
	def add_point(self, x_add, y_add):
		self.x = np.append(self.x, x_add)
		self.y = np.append(self.y, y_add)
		self.n += 1
		self.L = np.zeros((self.n, self.n), dtype = float)
		for i in range(self.n):
			self.L[i][0] = self.y[i]
			
	def Aitken_error_calculate(self, x_input, y_error):
		for i in range(self.n):
			self.L[i][0] = y_error
		for i in range(1, self.n):
			for j in range(i, self.n):
				self.L[j][i] = (self.L[j - 1][i - 1] * abs(self.x[j] - x_input) + self.L[j][i - 1] * abs(self.x[j - i] - x_input)) / abs(self.x[j] - self.x[j - i])
		for i in range(self.n):
			self.L[i][0] = self.y[i]
		return self.L[self.n - 1][self.n - 1]
	
def main():
	x = np.array([0, 1.75, 3.5, 5.25, 7])
	y = np.array([0, -1.307, -2.211, -0.927, -0.871])
	polynomial = Aitken(x, y, x.size)
	x_plot_table = np.linspace(0, 7, 50, dtype = float)
	y_plot_table = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_of_original = np.linspace(0, 0, 50, dtype = float)
	x = np.append(x, 2.555)
	y = np.append(y, polynomial.Aitken_calculate(2.555))
	for i in range(50):
		y_plot_table[i] = polynomial.Aitken_calculate(x_plot_table[i])
		y_plot_table_of_original[i] = math.cos(x_plot_table[i]) - 2 ** (0.1 * x_plot_table[i])
	fig, ax = plt.subplots()
	plt.plot(x_plot_table, y_plot_table, 'b-', label = 'Aitken')
	plt.plot(x_plot_table, y_plot_table_of_original, 'm--', label = 'Original')
	plt.plot(x, y, 'r*')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.annotate('f(x) ~ ' + str(y[y.size - 1]), xy=(x[x.size - 1], y[y.size - 1]), xytext=(x[x.size - 1], y[y.size - 1] - 0.32),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
	absolute_error = polynomial.Aitken_error_calculate(2.555, 0.0005)
	plt.text(1.0, 0.0, 'Abs. error = ' + str(absolute_error))
	plt.text(1.0, -0.1, 'Rel. error = ' + str(-absolute_error / y[y.size - 1]))
	plt.legend(loc='upper right')
	plt.show()
	
if __name__ == '__main__':
    main()