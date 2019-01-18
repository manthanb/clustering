import os
import numpy as np
import random
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


def calculate_mean(points, axis):
	s = 0
	l = 0
	for point in points: 
		s += point[axis]
		l += 1
	return s/l

def initialise_cluster_centers(input_data, k):

	cluster_centers = {}

	random_clusters = list([x for x in range(input_data.shape[0])])
	random.shuffle(random_clusters)

	for i in range(k):
		cluster_centers[i] = input_data[random_clusters[i], :]

	return cluster_centers

def assign_clusters(cluster_centers, input_data):

	cluster_assigned_data = {}
	
	for point in input_data:
		
		min_distance = math.pow(10,8)
		
		for index, center in cluster_centers.items():
			if index not in cluster_assigned_data:
				cluster_assigned_data[index] = []
			distance = np.linalg.norm(point - center)
			if distance < min_distance:
				min_distance = distance
				cluster_assigned_data[index].append(point)

	return cluster_assigned_data

def update_centers(cluster_centers, cluster_assigned_data):

	for index in cluster_assigned_data:
		m_x = calculate_mean(cluster_assigned_data[index], 0)
		m_y = calculate_mean(cluster_assigned_data[index], 1)
		cluster_centers[index] = np.array([m_x, m_y])

	return cluster_centers


def calculate_sse(cluster_centers, cluster_assigned_data):
	sse = 0
	for center in cluster_assigned_data:
		for point in cluster_assigned_data[center]:
			sse = sse + np.linalg.norm(point-cluster_centers[center])
	return sse


def kmeans(input_data, k, num_iterations):

	cluster_centers= initialise_cluster_centers(input_data, k)
	cluster_assigned_data = {}
	sse = []

	for i in range(num_iterations):
		cluster_assigned_data = assign_clusters(cluster_centers, input_data)
		cluster_centers = update_centers(cluster_centers, cluster_assigned_data)
		sse.append(calculate_sse(cluster_centers, cluster_assigned_data)/input_data.shape[0])
		visualise(cluster_centers, cluster_assigned_data, k)

	return cluster_centers, cluster_assigned_data, sse

def visualise(cluster_centers, cluster_assigned_data, k):

	colors = ['r', 'g', 'b','olive', 'teal', 'slategray', 'dodgerblue', 'brown', 'm', 'y']
	centroids = np.array([value for key, value in cluster_centers.items()])
	for i in range(k):
		points = np.array(cluster_assigned_data[i])
		plt.scatter(points[:, 0], points[:, 1], s=4, c=colors[i])
	plt.scatter(centroids[:,0], centroids[:,1], s=50, color='black', marker='^')
	plt.show()


def main():

	k = 3
	num_iterations = 14

	project_directory = os.getcwd()
	input_data = input(project_directory, 'Dataset_2.txt')
	cluster_centers, cluster_assigned_data, sse = kmeans(input_data, k, num_iterations)
	print(cluster_centers)
	plt.title('sse vs iterations for dataset-2. k=3. number of iterations = 14')
	plt.xlabel('iterations')
	plt.ylabel('sse')
	plt.plot(list([x for x in range(num_iterations)]), sse)
	plt.show()

main()
