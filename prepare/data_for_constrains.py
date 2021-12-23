import os

from ase.io import read, write
from ase import Atom, Atoms

import numpy as np

def get_atoms(inputfile,filetype):
	atoms = read(inputfile,format = filetype)
	return atoms

def min_distance(atoms):
	distances_matrix=atoms.get_all_distances(mic=True)
	m,n=distances_matrix.shape
	
	for i in range(m):
		for j in range(n):
			if i==j:
				distances_matrix[i,j]=100
	
	mind=distances_matrix.min()
	return mind

def get_min_distance_for_atoms(inputfile,filetype):
	atoms=get_atoms(inputfile,filetype)
	mind=min_distance(atoms)
	return mind

def get_min_distances_for_compounds(path,filetype,savepath,savename):#e.g.: dfc.get_min_distances_for_compounds(path='../data_for_distance/GAN_batch64_epoch70000_generated_crystal/',filetype='vasp',savepath='./',savename='train_y2.npy')
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	min_distances_list=[]

	filename=os.listdir(path)
	filename.sort()
	for i in range(len(filename)):
		print(i,filename[i])
		eachfile=filename[i]
		if eachfile.endswith(filetype):
			name=path+eachfile
			mind=get_min_distance_for_atoms(name,filetype)
			if mind<2:
				value=1
			else:
				value=0
			min_distances_list.append(value)
	print(sum(min_distances_list))
	min_distances_array=np.array(min_distances_list)
	np.save(savepath+savename,min_distances_array)


def get_min_distances_for_pre_compounds(path,filetype,savepath,savename):#e.g.: dfc.get_min_distances_for_compounds(path='../data_for_distance/GAN_batch64_epoch70000_generated_crystal/',filetype='vasp',savepath='./',savename='train_y2.npy')
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	min_distances_list=[]

	filename=os.listdir(path)
	filename.sort()
	for i in range(len(filename)):
		eachfile=filename[i]
		if eachfile.endswith(filetype):
			name=path+eachfile
			mind=get_min_distance_for_atoms(name,filetype)
			if mind<2:
				value=1
				min_distances_list.append(value)
			else:
				value=0
				command='rm '+path+eachfile
				os.system(command)
	print(sum(min_distances_list))
	min_distances_array=np.array(min_distances_list)
	np.save(savepath+savename,min_distances_array)

def get_statistic_file(directory,standard_delta,savepath):
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	min_distance_array=np.load(directory)
	min_value=np.floor(min_distance_array.min())
	max_value=np.ceil(min_distance_array.max())


	number_of_bins=int((max_value-min_value)/standard_delta)
	statistics=np.zeros([number_of_bins,2])
	for j in range(number_of_bins):
		statistics[j,1]=min_value+j*standard_delta
	for i in min_distance_array:
		for j in range(number_of_bins):
			if i>=min_value+j*standard_delta and i<min_value+(j+1)*standard_delta:
				statistics[j,0]=statistics[j,0]+1
	print(statistics)
	np.savetxt(savepath+"distance_statistics.csv", statistics, delimiter=",")
	
def get_formation_energy_statistics(data_path,standard_delta,savepath):
	filename=os.listdir(data_path)
	formation_energy_list=[]
	for eachfilename in filename:
		formation_energy=np.load(data_path+eachfilename)
		formation_energy_list.append(formation_energy)
	formation_energy_array=np.array(formation_energy_list)
	min_value=np.floor(formation_energy_array.min())
	max_value=np.ceil(formation_energy_array.max())

	number_of_bins=int((max_value-min_value)/standard_delta)
	statistics=np.zeros([number_of_bins,2])
	for j in range(number_of_bins):
		statistics[j,1]=min_value+j*standard_delta
	for i in formation_energy_array:
		for j in range(number_of_bins):
			if i>=min_value+j*standard_delta and i<min_value+(j+1)*standard_delta:
				statistics[j,0]=statistics[j,0]+1
	print(statistics)
	np.savetxt(savepath+"distance_statistics.csv", statistics, delimiter=",")
			

def get_formation_energy_constrain(data_path,constrain_path):
	if not os.path.exists(constrain_path):
		os.makedirs(constrain_path)
	filename=os.listdir(data_path)
	filename.sort()
	for eachfilename in filename:
		formation_energy=np.load(data_path+eachfilename)
		if formation_energy<0:
			unstable=0
		else:
			unstable=1
		np.save(constrain_path+eachfilename,unstable)

def get_convex_hull_constrain(data_path,constrain_path):
	if not os.path.exists(constrain_path):
		os.makedirs(constrain_path)
	filename=os.listdir(data_path)
	filename.sort()
	for eachfilename in filename:
		convex_hull=np.load(data_path+eachfilename)
		if convex_hull<0.15:
			unstable=0
		else:
			unstable=1
		np.save(constrain_path+eachfilename,unstable)

def get_formation_energy_constrain_train_y(data_path,directory):#e.g.: get_formation_energy_constrain_train_y(data_path='./formation_energy/',directory='./train_y.npy')
	filename=os.listdir(data_path)
	filename.sort()
	train_y=[]
	for i in range(len(filename)):
		eachnpyfile=filename[i]
		if eachnpyfile.endswith('.npy'):
			formation_energy=np.load(data_path+eachnpyfile)
			if formation_energy<0:
				unstable=0
			else:
				unstable=1
			train_y.append(unstable)
	print("number of non statble:", sum(train_y))
	np.save(directory,np.array(train_y))

def get_formation_energy_constrain_reg_train_y(data_path,directory):#e.g.: get_formation_energy_constrain_train_y(data_path='./formation_energy/',directory='./train_y.npy')
	filename=os.listdir(data_path)
	filename.sort()
	train_y=[]
	for i in range(len(filename)):
		eachnpyfile=filename[i]
		if eachnpyfile.endswith('.npy'):
			formation_energy=np.load(data_path+eachnpyfile)

			train_y.append(formation_energy)
	np.save(directory,np.array(train_y))

def get_convex_hull_constrain_train_y(data_path,directory):#e.g.: get_convex_hull_constrain_train_y(data_path='./convex_hull/',directory='./train_y.npy')
	filename=os.listdir(data_path)
	filename.sort()
	train_y=[]
	for i in range(len(filename)):
		eachnpyfile=filename[i]
		if eachnpyfile.endswith('.npy'):
			convex_hull=np.load(data_path+eachnpyfile)
			if convex_hull<0.15:
				unstable=0
			else:
				unstable=1
			train_y.append(unstable)
	print("number of non statble:", sum(train_y))
	np.save(directory,np.array(train_y))


