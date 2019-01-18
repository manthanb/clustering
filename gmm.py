import os
import numpy as np
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt

def load_text_file(file_path):
	file = open(file_path, 'r')
	return file.readlines()


def input(dirpath,file_name):
	input_file = load_text_file(dirpath+'/'+file_name)
	input_data = []
	for line in input_file:
		tokens = line.split()
		current_coordinates = [float(tokens[0]), float(tokens[1])]
		input_data.append(current_coordinates)
	return np.array(input_data)


def initialise_means(input_data, k):
	
	mu = {}
	max_x, max_y = np.max(input_data, axis=0)[0], np.max(input_data, axis=0)[1]
	
	for i in range(k):
		mu_x, mu_y = (-max_x) * np.random.random_sample() + max_x, (-max_y) * np.random.random_sample() + max_y
		mu[i] = np.array([mu_x, mu_y])
	
	return mu


def initialise_covariance_matrix(input_data, k):
	
	sigma = {}

	for i in range(k):
		gamma = np.random.random_sample()
		s = np.cov(input_data.T)
		s = gamma * s
		sigma[i] = s

	return sigma


def generate_pi(k):
	pi = np.random.randint(low=0, high=10, size=(k,))
	pi = pi / np.sum(pi)
	return pi


def calculate_covariance_matrix(input_data, mean):
	sigma = np.dot((input_data-mean).T, (input_data-mean))
	return sigma


def expectation(input_data, mu, sigma, pi):

	gamma = np.zeros((input_data.shape[0], len(mu)))

	for i in range(input_data.shape[0]):
		denominator = 0
		for j in range(len(mu)):
			x_n = input_data[i,:]
			mu_k = mu[j]
			sigma_k = sigma[j]
			pi_k = pi[j]
			numerator = pi_k * multivariate_normal.pdf(x_n, mean=mu_k, cov=sigma_k)
			denominator = denominator + numerator
			gamma[i,j] = numerator
		gamma[i,:] = gamma[i,:] / denominator

	return gamma


def maximization(input_data, gamma):

	mu, sigma = {}, {}

	N_k = np.sum(gamma, axis=0)

	for i in range(gamma.shape[1]):
		s = 0
		for j in range(input_data.shape[0]):
			s = s + (gamma[j,i] * input_data[j,:])
		mu[i] = s / N_k[i]

	for i in range(gamma.shape[1]):
		s = 0
		for j in range(input_data.shape[0]):
			cov = np.dot((input_data[j,:]-mu[i]).reshape(2,1),(input_data[j,:]-mu[i]).reshape(2,1).T)
			s = s + (gamma[j,i] * cov)
		sigma[i] = s / N_k[i]	
			
	pi = N_k / input_data.shape[0]

	return mu, sigma, pi


def calculate_log_likelihood(input_data, pi, mu, sigma):
	
	ll = 0
	
	for i in range(input_data.shape[0]):
		s = 0
		x_n = input_data[i,:]
		for j in range(len(mu)):
			mu_k = mu[j]
			sigma_k = sigma[j]
			pi_k = pi[j]
			s = s + pi_k * multivariate_normal.pdf(x_n, mean=mu_k, cov=sigma_k)
		ll = ll + math.log(s)

	return ll


def estimate(input_data, mu, sigma, pi, num_iter):
	ll = []
	for i in range(num_iter):
		# print(mu)
		gamma = expectation(input_data, mu, sigma, pi)
		mu, sigma, pi = maximization(input_data, gamma)
		print(mu)
		# ll.append(calculate_log_likelihood(input_data, pi, mu, sigma))
	return mu, sigma, pi, gamma, ll


def main():

	project_directory = os.getcwd()
	input_data = input(project_directory, 'Dataset_2.txt')
	k = 3
	num_iter = 50

	mu = {}

	# mu[0] = np.array([ 1.45290602, -0.00342932])
	# mu[1] = np.array([-0.56156164, -0.07088148])

	mu[0] = np.array([0.6145112 , 0.05597526])
	mu[1] = np.array([-0.96118352,  0.05081451])
	mu[2] = np.array([1.9295404 , 0.18171246])

	# mu = initialise_means(input_data, k)
	sigma = initialise_covariance_matrix(input_data, k)
	pi = generate_pi(k)

	mu, sigma, pi, gamma, ll = estimate(input_data, mu, sigma, pi, num_iter)
	
	assignment = []
	for g in gamma:
		print(g)
		assignment.append(np.argmax(g))

	colors = ['r', 'g', 'b']
	for i in range(input_data.shape[0]):
		plt.title('Dataset-1 with kmeans initialised means. k=2')
		plt.scatter(input_data[i,0], input_data[i,1], s=4, c=colors[assignment[i]])

	# plt.plot(list([x for x in range(num_iter)]),ll)
	# plt.title('Dataset-1 with random means')
	# plt.xlabel('number of iterations')
	# plt.ylabel('log likelihood')
	plt.show()
	
	
main()