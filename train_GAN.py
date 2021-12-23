import prepare.data_transformation as dt
import prepare.generate_train as gt

GAN_database_folder_path='./database/'
GAN_calculation_folder_path='./calculation/'

##### 1. generate lattice and sites graph

#### 1.1. generate 3d lattice voxel graph
print('generating 3d lattice voxel graph')
dt.generate_lattice_graph(lattice_graph_path=GAN_calculation_folder_path+'original_lattice_graph/',atomlisttype='specified',a_list=['Bi'],data_path=GAN_database_folder_path+'geometries/',data_type='vasp')

#### 1.2. generate 3d sites voxel graph (this graph will generate for each element. For binary it is a list file with length to be 2)
print('generating 3d sites voxel graph')
dt.generate_sites_graph(sites_graph_path=GAN_calculation_folder_path+'original_sites_graph/',atomlisttype='specified',a_list=['Bi','Se'],data_path=GAN_database_folder_path+'geometries/',data_type='vasp')


##### 2. train the autoencoder for 3d voxel graph and generate encoded lattice and sites of the voxel graphs

#### 2.1. generate encoded lattice and save the trained model
print('lattice autoencoder running:')
import prepare.lattice_autoencoder as la #comment: lattice_autoencoder has many similar functions as sites_autoencoder, so need to be imported every time used
la.lattice_autocoder(lattice_graph_path=GAN_calculation_folder_path+'original_lattice_graph/',encoded_graph_path=GAN_calculation_folder_path+'original_encoded_lattice/',model_path=GAN_calculation_folder_path+'model/')

#### 2.2. generate encoded sites and save the trained model (different elements are trained together since this is only a encoding step)
print('sites autoencoder running:')
import prepare.sites_autoencoder as sa
sa.sites_autocoder(sites_graph_path=GAN_calculation_folder_path+'original_sites_graph/',encoded_graph_path=GAN_calculation_folder_path+'original_encoded_sites/',model_path=GAN_calculation_folder_path+'model/')


##### 3. generate 2d graph

#### 3.1. directly combine the lattice and sites together
print('generating 2d graphs')
dt.generate_crystal_2d_graph(encodedgraphsavepath=GAN_calculation_folder_path+'original_encoded_sites/',encodedlatticesavepath=GAN_calculation_folder_path+'original_encoded_lattice/',crystal_2d_graph_path=GAN_calculation_folder_path+'original_crystal_2d_graphs/')

#### 3.2. transform the 2d graph into square shape and combine them in one file in order to generate the training file for GAN
print('converting 2d graphs into square shape and combining all the training data')
gt.generate_train_X(encodedsavepath=GAN_calculation_folder_path+'original_crystal_2d_graphs/',X_train_savepath=GAN_calculation_folder_path,X_train_name='train_X.npy')

command='rm -r ./calculation/original_encoded_sites/;'
os.system(command)
command='rm -r ./calculation/original_encoded_lattice/;'
os.system(command)
command='rm -r ./calculation/original_sites_graph/;'
os.system(command)
command='rm -r ./calculation/original_lattice_graph/;'
os.system(command)

##### 4. train constrain

#### 4.1. parepare data to train formation energy reg network
import prepare.data_for_constrains as dfc
print('generate formation energy reg constrain data')
dfc.get_formation_energy_constrain_reg_train_y(data_path=GAN_database_folder_path+'properties/formation_energy/',directory=GAN_calculation_folder_path+'train_formation_energy_reg.npy')

#### 4.2.train the formation energy reg network
import prepare.constrain_reg as con_reg
print('train formation energy reg constrain')
constrain = con_reg.constrain()
constrain.train(X_npy=GAN_calculation_folder_path+'train_X.npy',y_npy=GAN_calculation_folder_path+'train_formation_energy_reg.npy',model_path=GAN_calculation_folder_path+'model/formation_energy_reg.h5',epochs=20001, batch_size=64, save_interval=10000)

##### 5. train CCDCGAN model
import gan.ccdcgan as gan
ccdcgan = gan.CCDCGAN()
ccdcgan.train(epochs = 1000000，batch_size=128, save_interval=5000,GAN_calculation_folder_path=GAN_calculation_folder_path,X_train_name='train_X.npy')

print（'the training process has finished'）