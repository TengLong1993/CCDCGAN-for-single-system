import numpy as np
import os
from ase.io import read,write

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

def combine_for_pre(generated_pre_lattice_path='./generated_lattice/',generated_pre_sites_path='./generated_sites/',generated_crystal_path='./generated_crystal/'):
	if not os.path.exists(generated_crystal_path):
		os.makedirs(generated_crystal_path)

	filename=os.listdir(generated_pre_lattice_path)
	for eachfile in filename:
		if eachfile.endswith('.vasp'):
			print(eachfile)
			try:
				filename=generated_pre_lattice_path+eachfile
				cell=read(filename)
				cg_cell = cell.get_positions()

				filename=generated_pre_sites_path+eachfile
				real_mat=read(filename)
				pos=real_mat.get_positions()

				delta=cg_cell - np.mean(pos,0)
				new_pos=pos+delta

				real_mat.set_cell(cell.get_cell())
				real_mat.set_positions(new_pos)

				write(generated_crystal_path+eachfile,real_mat)
			except:
				print(eachfile+' is not working')
				continue

def combine(generated_pre_lattice_path='./generated_lattice/',generated_pre_sites_path='./generated_sites/',generated_crystal_path='./generated_crystal/'):
	if not os.path.exists(generated_crystal_path):
		os.makedirs(generated_crystal_path)

	filename=os.listdir(generated_pre_lattice_path)
	for eachfile in filename:
		if eachfile.endswith('.vasp'):
			try:
				filename=generated_pre_lattice_path+eachfile
				cell=read(filename)
				cg_cell = cell.get_positions()

				filename=generated_pre_sites_path+eachfile
				real_mat=read(filename)
				pos=real_mat.get_positions()

				delta=cg_cell - np.mean(pos,0)
				new_pos=pos+delta

				real_mat.set_cell(cell.get_cell())
				real_mat.set_positions(new_pos)

				write(generated_crystal_path+eachfile,real_mat,format='vasp')
				real_mat=read(generated_crystal_path+eachfile)
				mind=get_min_distance_for_atoms(generated_crystal_path+eachfile,filetype='vasp')

				if mind<0.5:
					print('does not fulfill distance constrain')
				else:
					write(generated_crystal_path+eachfile[:-5]+'.cif',real_mat,format='cif')

				command='rm '+ generated_crystal_path+eachfile
				os.system(command)
			except:
				print('is not working')
				continue

