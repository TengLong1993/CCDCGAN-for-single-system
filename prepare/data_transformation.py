import os

from ase.io import read, write
from ase import Atom, Atoms

import numpy as np
from numpy.linalg import norm
import random

from joblib import Parallel, delayed

from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

import math

#####define the used atom list
def get_atomlist_atomindex(atomlisttype='all element',a_list=None):#atomlisttype indicate which kind of list to be used; 'all element' is to use the whole peroidic table, 'specified' is to give a list on our own
	if atomlisttype=='all element':
		all_atomlist = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
		          'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
		          'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
		          'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
		          'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
		          'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
		          'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
	else:
		if atomlisttype=='specified':
			all_atomlist=a_list
		else:
			print('atom list type is not acceptable')
			return
	cod_atomlist=all_atomlist
	cod_atomindex = {}
	for i,symbol in enumerate(all_atomlist):
		cod_atomindex[symbol] = i

	return cod_atomlist,cod_atomindex

def get_scale(sigma):
	scale = 1.0/(2*sigma**2)
	return scale

def get_atoms(inputfile,filetype):
	atoms = read(inputfile,format = filetype)
	return atoms

def extract_cell(atoms):
	cell = atoms.cell
	atoms_ = Atoms('Bi')
	atoms_.cell = cell
	atoms_.set_scaled_positions([0.5,0.5,0.5])
	return atoms_

def get_fakeatoms_grid(atoms,nbins):
	atomss = []
	scaled_positions = []
	ijks = []
	grid = np.array([float(i)/float(nbins) for i in range(nbins)])
	yv,xv,zv = np.meshgrid(grid,grid,grid)
	pos = np.zeros((nbins**3,3))
	pos[:,0] = xv.flatten()
	pos[:,1] = yv.flatten()
	pos[:,2] = zv.flatten()
	atomss = Atoms('H'+str(nbins**3))
	atomss.set_cell(atoms.get_cell())#making pseudo-crystal containing H positioned at pre-defined fractional coordinate
	atomss.set_pbc(True)
	atomss.set_scaled_positions(pos)
	fakeatoms_grid = atomss
	return fakeatoms_grid

def get_image_one_atom(atom,fakeatoms_grid,nbins,scale):
	grid_copy = fakeatoms_grid.copy()
	ngrid = len(grid_copy)
	image = np.zeros((1,nbins**3))
	grid_copy.append(atom)
	drijk = grid_copy.get_distances(-1,range(0,nbins**3),mic=True)
	pijk = np.exp(-scale*drijk**2)
	image[:,:] = pijk.flatten()
	return image.reshape(nbins,nbins,nbins)

def get_image_all_atoms(atoms,nbins,scale,norm,num_cores,atomlisttype,a_list):
	fakeatoms_grid = get_fakeatoms_grid(atoms,nbins)
	cell = atoms.get_cell()
	imageall_gen = Parallel(n_jobs=num_cores)(delayed(get_image_one_atom)(atom,fakeatoms_grid,nbins,scale) for atom in atoms)
	imageall_list = list(imageall_gen)
	cod_atomlist,cod_atomindex = get_atomlist_atomindex(atomlisttype,a_list)
	nchannel = len(cod_atomlist)
	channellist = []
	for i,atom in enumerate(atoms):
		channel = cod_atomindex[atom.symbol]
		channellist.append(channel)
	channellist = list(set(channellist))
	nc = len(channellist)
	shape = (nbins,nbins,nbins,nc)
	image = np.zeros(shape,dtype=np.float32)
	for i,atom in enumerate(atoms):
		nnc = channellist.index(cod_atomindex[atom.symbol])
		img_i = imageall_list[i]
		image[:,:,:,nnc] += img_i * (img_i>=0.02)
		 
	return image,channellist


def basis_translate(atoms):
	N = len(atoms)
	pos = atoms.positions
	cg = np.mean(pos,0)
	dr = 7.5 - cg #move to center of 15A-cubic box
	dpos = np.repeat(dr.reshape(1,3),N,0)
	new_pos = dpos + pos
	atoms_ = atoms.copy()
	atoms_.cell = 15.0*np.identity(3)
	atoms_.positions = new_pos
	return atoms_

def generate_sites_graph(sites_graph_path,atomlisttype,a_list,data_path,data_type):#e.g.: dt.generate_sites_graph(sites_graph_path='./original_lattice_graph/',atomlisttype='specified',a_list=['V','O'],data_path='/home/teng/tensorflow2.0_example/imatgen-master/iMatGen-VO_dataset_generated_strctures/VO_dataset/geometries/',data_type='vasp')
	if not os.path.exists(sites_graph_path):
		os.makedirs(sites_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)#'./TC/chem_info/')

	for eachfile in filename:
		if eachfile.endswith(data_type):
			filename=data_path+eachfile
			atoms=get_atoms(filename,data_type)
			atoms_=basis_translate(atoms)
			image,channellist=get_image_all_atoms(atoms_,64,scale,norm,8,atomlisttype,a_list)

			savefilename=sites_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,image)

def generate_combined_sites_graph(sites_graph_path,atomlisttype,a_list,data_path,data_type):
	if not os.path.exists(sites_graph_path):
		os.makedirs(sites_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)

	for eachfile in filename:
		if eachfile.endswith(data_type):
			filename=data_path+eachfile
			print(filename)
			atoms=get_atoms(filename,data_type)
			atoms_=basis_translate(atoms)
			image,channellist=get_image_all_atoms(atoms_,64,scale,norm,8,atomlisttype,a_list)

			_,_,_,nc=image.shape
			combined_image=np.zeros([64,64,64])
			for i in range(nc):
				combined_image=combined_image+image[:,:,:,i]

			savefilename=sites_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,combined_image)

def generate_lattice_graph(lattice_graph_path,atomlisttype,a_list,data_path,data_type):#e.g.: dt.generate_lattice_graph(lattice_graph_path='./original_lattice_graph/',atomlisttype='specified',a_list=['V'],data_path='/home/teng/tensorflow2.0_example/imatgen-master/iMatGen-VO_dataset_generated_strctures/VO_dataset/geometries/',data_type='vasp')
	if not os.path.exists(lattice_graph_path):
		os.makedirs(lattice_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)
    
	for eachfile in filename:
		if eachfile.endswith(data_type):

			filename=data_path+eachfile
			print(filename)
			atoms=get_atoms(filename,data_type)
			atoms_=extract_cell(atoms)
			image,channellist=get_image_all_atoms(atoms_,32,scale,norm,8,atomlisttype,a_list)
			image=image.reshape(32,32,32)

			savefilename=lattice_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,image)

def generate_crystal_2d_graph(encodedgraphsavepath='./encoded_sites/',encodedlatticesavepath='./encoded_lattice/',crystal_2d_graph_path='./crystal_2d_graphs/'):#e.g.: dt.generate_crystal_2d_graph(encodedgraphsavepath='./original_encoded_sites/',encodedlatticesavepath='./original_encoded_lattice/',crystal_2d_graph_path='./original_crystal_2d_graphs/')
	if not os.path.exists(crystal_2d_graph_path):
		os.makedirs(crystal_2d_graph_path)
	filename=os.listdir(encodedlatticesavepath)
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			encodeddirectory=encodedlatticesavepath+eachnpyfile
			crystal_2d_graph=np.zeros([6,200])
			encoded_lattice=np.load(encodeddirectory)
			crystal_2d_graph[0,:]=encoded_lattice.reshape(200)
			encodeddirectory=encodedgraphsavepath+eachnpyfile
			for i in range(1,3):
				encoded_sites=np.load(encodeddirectory)[:,i-1]
				crystal_2d_graph[i,:]=encoded_sites.reshape(200)
			savefilename=crystal_2d_graph_path+eachnpyfile
			np.save(savefilename,crystal_2d_graph)
			for i in range(200):
				if crystal_2d_graph[0,i]!=encoded_lattice[i]:
					print(crystal_2d_graph[0,i],encoded_lattice[i])
					exit()

def change_lattice_in_crystal_2d_graph(previous_crystal_2d_graph_path='./crystal_2d_graphs/',encodedlatticesavepath='./encoded_lattice/',crystal_2d_graph_path='./crystal_2d_graphs/'):
	if not os.path.exists(crystal_2d_graph_path):
		os.makedirs(crystal_2d_graph_path)
	filename=os.listdir(encodedlatticesavepath)
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			encodeddirectory=encodedlatticesavepath+eachnpyfile
			crystal_2d_graph=np.load(previous_crystal_2d_graph_path+eachnpyfile)#np.zeros([6,200])
			encoded_lattice=np.load(encodeddirectory)
			crystal_2d_graph[0,:]=encoded_lattice.reshape(200)
			savefilename=crystal_2d_graph_path+eachnpyfile
			np.save(savefilename,crystal_2d_graph)
			for i in range(200):
				if crystal_2d_graph[0,i]!=encoded_lattice[i]:
					print(crystal_2d_graph[0,i],encoded_lattice[i])
					exit()

def detect_peaks(image):
	neighborhood = generate_binary_structure(3, 2)
	local_max = (maximum_filter(image, footprint = neighborhood, mode = "wrap") == image)

	background = (image < 0.02)

	eroded_background = binary_erosion(background, structure = neighborhood, border_value = 1)
	detected_peaks = np.logical_and(local_max, np.logical_not(eroded_background))
	return detected_peaks

def reconstruction(image,ele):
	# image should have dimension of (N,N,N)
	image0 = gaussian_filter(image,sigma=0.15)
	peaks = detect_peaks(image0)

	recon_mat = Atoms(cell=15*np.identity(3),pbc=[1,1,1])
	(peak_x,peak_y,peak_z) = np.where(peaks==1.0)
	for px,py,pz in zip(peak_x,peak_y,peak_z):
		if np.sum(image[px-3:px+4,py-3:py+4,pz-3:pz+4] > 0) >= 0:
			recon_mat.append(Atom(ele,(px/64.0,py/64.0,pz/64.0)))

	pos = recon_mat.get_positions()
	recon_mat.set_scaled_positions(pos)

	return recon_mat

def generated_sites(genenrated_decoded_path='./generated_decoded_sites/',generated_pre_path='./generated_sites/',element_list=['Fe','Co']):
	if not os.path.exists(generated_pre_path):
		os.makedirs(generated_pre_path)

	filename=os.listdir(genenrated_decoded_path)
	bad_reproduce=0
	ele = element_list
	for eachfile in filename:
		if eachfile.endswith('.npy'):
			filename=genenrated_decoded_path+eachfile
			img=np.load(filename)

			tmp_mat = []
			for idc in range(2):
				image = img[:,:,:,idc].reshape(64,64,64)
				tmp_mat.append(reconstruction(image,ele[idc]))
			for atom in tmp_mat[-1]:
				tmp_mat[0].append(atom)	

			try:
				write(generated_pre_path+eachfile[:-4]+'.vasp',tmp_mat[0])
			except:
				bad_reproduce=bad_reproduce+1
				continue
	print(bad_reproduce)

def compute_length(axis_val):
	non_zeros = axis_val[axis_val > 0]
	
	(a,) = np.where(axis_val == non_zeros.min())

	# distance from center in grid space
	N = np.abs(16 - a[0])
	
	# length of the unit vector
	r_fake = np.sqrt(-2*0.26**2*np.log(non_zeros.min())) #r_fake = N*(r/32)
	r = r_fake * 32.0 / float(N)
	return r
	
def compute_angle(ri,rj,rij):
	cos_theta = (ri**2 + rj**2 - rij**2) / (2*ri*rj)
	theta = math.acos(-cos_theta) * 180/np.pi # angle in deg.
	return theta 

def get_atoms(inputfile,filetype):
	atoms = read(inputfile,format = filetype)
	return atoms

def generated_lattice(genenrated_decoded_path='./generated_decoded_lattice/',generated_pre_path='./generated_lattice/'):
	if not os.path.exists(generated_pre_path):
		os.makedirs(generated_pre_path)

	filename=os.listdir(genenrated_decoded_path)
	bad_reproduce=0
	for eachfile in filename:
		if eachfile.endswith('.npy'):

			filename=genenrated_decoded_path+eachfile
			img=np.load(filename)

			a_axis = img[:,16,16]
			try:
				ra = compute_length(a_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			b_axis = img[16,:,16]
			try:
				rb = compute_length(b_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			c_axis = img[16,16,:]
			try:
				rc = compute_length(c_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				ab_axis = np.array([img[i,i,16] for i in range(32)]); rab = compute_length(ab_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				bc_axis = np.array([img[16,i,i] for i in range(32)]); rbc = compute_length(bc_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				ca_axis = np.array([img[i,16,i] for i in range(32)]); rca = compute_length(ca_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue

			try:
				alpha = compute_angle(rb,rc,rbc)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				beta = compute_angle(rc,ra,rca)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				gamma = compute_angle(ra,rb,rab)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue

			try:
				atoms = Atoms(cell=[ra,rb,rc,alpha,beta,gamma],pbc=True)
			except AssertionError:
				bad_reproduce=bad_reproduce+1
				continue
			atoms.append(Atom('Cu',[0.5]*3))
			pos = atoms.get_positions()
			atoms.set_scaled_positions(pos)
			try:
				write(generated_pre_path+eachfile[:-4]+'.vasp',atoms)
			except RuntimeError:
				bad_reproduce=bad_reproduce+1
				continue
	print(bad_reproduce)

#####train test generate
def train_test_split(path='./3d_crystal_graphs/',split_ratio=0.2):
	filename=os.listdir(path)
	name_list=[]
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			name_list.append(eachnpyfile[:-4])
	test_size=round(split_ratio*len(name_list))
	random.shuffle(name_list)
	train_name_list=name_list[test_size:]
	test_name_list=name_list[:test_size]
	return test_size,test_name_list,train_name_list

#####get batch list
def get_batch_name_list(train_name_list,batch_size=24):
	random.shuffle(train_name_list)
	batch_name_list=train_name_list[:batch_size]
	return batch_name_list

#####get batch lattice input
def generate_lattice_batch(batch_size,latticesavepath='./lattice/',name_list=['mp-1183837_Co3Ni']):
	batch_lattices=np.zeros([batch_size,32,32,32,1])
	for i in range(0,batch_size):
		batch_lattices[i,:,:,:,:]=read_lattice(latticesavepath,name_list[i]).reshape([1,32,32,32,1])
	return batch_lattices

#####get batch element input
def generate_graph_batch(batch_size,graphsavepath='./3d_crystal_graphs/',name_list=['mp-1183837_Co3Ni'],element=0):
	batch_three_d_graphs=np.zeros([batch_size,64,64,64,2])
	for i in range(0,batch_size):
		batch_three_d_graphs[i,:,:,:,:]=read_crystal_graph(graphsavepath,name_list[i]).reshape([1,64,64,64,2])
		batch_three_d_graphs=batch_three_d_graphs[:,:,:,:,element].reshape([1,64,64,64,1])
	return batch_three_d_graphs

#####get batch 2d graph input
def generate_2dgraph_batch(batch_size,graph2d_savepath='./crystal_2d_graphs/',name_list=['mp-1183837_Co3Ni']):
	batch_2d_graphs=np.zeros([batch_size,1,200,6])
	for i in range(0,batch_size):
		batch_2d_graphs[i,:,:,:]=read_crystal_graph(graph2d_savepath,name_list[i]).reshape([1,1,200,6])
	return batch_2d_graphs

#####get 3d graph
def read_crystal_graph(graphsavepath='./3d_crystal_graphs/',name='mp-1183837_Co3Ni'):
	filename=graphsavepath+name+'.npy'
	three_d_graph=np.load(filename)
	return three_d_graph

#####get lattice
def read_lattice(latticesavepath='./lattice/',name='mp-1183837_Co3Ni'):
	filename=latticesavepath+name+'.npy'
	lattice=np.load(filename)
	return lattice
