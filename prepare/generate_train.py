import numpy as np
import os

def combine_two_database(database1_path='./a.npy',database2_path='./b.npy',new_database_path='./c.npy'):
	database1=np.load(database1_path)
	database2=np.load(database2_path)
	database_new=np.append(database1, database2, axis=0)
	np.save(new_database_path,database_new)


def generate_train_X(encodedsavepath='/home/teng/tensorflow2.0_example/ccGAN/original_crystal_2d_graphs_new/',X_train_savepath='./',X_train_name='train_X.npy'):

	filename=os.listdir(encodedsavepath)
	filename.sort()
	train_X=int(0)
	for i in range(len(filename)):
		eachnpyfile=filename[i]
		if eachnpyfile.endswith('.npy'):
			directory=encodedsavepath+eachnpyfile
			crystal_2d_graph=np.load(directory)
			crystal_2d_graph_new=np.zeros(28*28)
			for i in [0,1,2]:
				crystal_2d_graph_new[i*200:i*200+200]=crystal_2d_graph[i,:]
			crystal_2d_graph_new=crystal_2d_graph_new.reshape(1,28,28)
			if type(train_X)==int:
				train_X=crystal_2d_graph_new
			else:
				train_X=np.append(train_X, crystal_2d_graph_new, axis=0)

	np.save(X_train_savepath+X_train_name,train_X)

def generate_pre_train_X(crystal_folder_path='pre_crystal_for_check/',encodedsavepath='pre_2d_graph_for_check/',X_train_savepath='./',X_train_name='distance_train_X.npy'):

	filename=os.listdir(crystal_folder_path)
	filename.sort()
	train_X=int(0)
	for i in range(len(filename)):
		eachnpyfile=filename[i]
		if eachnpyfile.endswith('.vasp'):
			directory=encodedsavepath+eachnpyfile[:-5]+'.npy'
			crystal_2d_graph=np.load(directory)
			crystal_2d_graph_new=np.zeros(28*28)
			for i in [0,1,2]:
				crystal_2d_graph_new[i*200:i*200+200]=crystal_2d_graph[i,:]
			crystal_2d_graph_new=crystal_2d_graph_new.reshape(1,28,28)
			if type(train_X)==int:
				train_X=crystal_2d_graph_new
			else:
				train_X=np.append(train_X, crystal_2d_graph_new, axis=0)

	np.save(X_train_savepath+X_train_name,train_X)

def generate_train_y(constrainsavepath='/home/teng/tensorflow2.0_example/ccGAN/formation_energy_constrain/',y_train_savepath='./',y_train_name='train_y.npy'):
	
	filename=os.listdir(constrainsavepath)
	train_y=[]
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			directory=constrainsavepath+eachnpyfile
			constrain=np.load(directory)
			train_y.append(constrain)
	train_y=np.array(train_y)
	np.save(y_train_savepath+y_train_name,train_y)

def generate_X_of_train_y2(constrainsavepath='/home/teng/tensorflow2.0_example/data_for_distance/GAN_batch64_epoch70000_generated_crystal/',Xsavepath='/home/teng/tensorflow2.0_example/data_for_distance/GAN_batch64_epoch70000_generated_2d_graphs/'):
	
	filename=os.listdir(constrainsavepath)
	train_X=int(0)
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.vasp'):
			directory=Xsavepath+eachnpyfile[:-5]+'.npy'
			crystal_2d_graph=np.load(directory)
			crystal_2d_graph_new=np.zeros(28*28)
			for i in [0,1,2]:
				crystal_2d_graph_new[i*200:i*200+200]=crystal_2d_graph[i,:]
			crystal_2d_graph_new=crystal_2d_graph_new.reshape(1,28,28)
			if type(train_X)==int:
				train_X=crystal_2d_graph_new
			else:
				train_X=np.append(train_X, crystal_2d_graph_new, axis=0)

	np.save('./train_X2.npy',train_X)

