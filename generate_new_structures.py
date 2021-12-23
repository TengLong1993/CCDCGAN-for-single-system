import prepare.data_transformation as dt
import generation.combine_cell_basis as ccb

GAN_database_folder_path='./database/'
GAN_calculation_folder_path='./calculation/'

##### 6. generate new 2d graphs
print('generating 2d graphs')
import generation.predict_dcgan as gan
dcgan = gan.DCGAN()
dcgan.predict(epochs=10000, GAN_calculation_folder_path=GAN_calculation_folder_path)

##### 7. check gemerated graphs
print('checking generated graphs')
import numpy as np
import os
directory='./calculation/generated_2d_graph_square/'
save_directory='./calculation/generated_2d_graph/'
if not os.path.exists(save_directory):
 os.makedirs(save_directory)
filename=os.listdir(directory)
for eachnpyfile in filename:
 if eachnpyfile.endswith('.npy'):
  data_single=np.load(directory+eachnpyfile).reshape(28*28)
  data_single_original=np.zeros([6,200])
  for j in [0,1,2]:
   data_single_original[j,:]=data_single[200*j:200*j+200]
  np.save(save_directory+eachnpyfile,data_single_original)


##### 8. generate some new structures fulfill the requirements
#### 8.1. restore voxel lattice
print('restore generated voxel lattice')
import prepare.lattice_autoencoder as la
la.lattice_restorer(generated_2d_path=GAN_calculation_folder_path+'generated_2d_graph/',genenrated_decoded_path=GAN_calculation_folder_path+'generated_decoded_lattice_for_check/',model_path=GAN_calculation_folder_path+'model/')

#### 8.2. restore voxel sites
print('restore generated voxel sites')
import prepare.sites_autoencoder as sa
sa.sites_restorer(generated_2d_path=GAN_calculation_folder_path+'generated_2d_graph/',genenrated_decoded_path=GAN_calculation_folder_path+'generated_decoded_sites_for_check/',model_path=GAN_calculation_folder_path+'model/')

#### 8.3. restore real lattice
print('restore generated real lattice')
dt.generated_lattice(genenrated_decoded_path=GAN_calculation_folder_path+'generated_decoded_lattice_for_check/',generated_pre_path=GAN_calculation_folder_path+'generated_real_lattice_for_check/')

#### 8.4. restore real sites
print('restore generated real sites')
dt.generated_sites(genenrated_decoded_path=GAN_calculation_folder_path+'generated_decoded_sites_for_check/',generated_pre_path=GAN_calculation_folder_path+'generated_real_sites_for_check/',element_list=['Co','Hf'])

#### 8.5. combine and generate ".cif"
print('generate .cif')
ccb.combine(generated_pre_lattice_path=GAN_calculation_folder_path+'generated_real_lattice_for_check/',generated_pre_sites_path=GAN_calculation_folder_path+'generated_real_sites_for_check/',generated_crystal_path=GAN_calculation_folder_path+'generated_crystal_for_check/')


##### 9. copy data
print('copying data')
if not os.path.exists('./save/generated_2d_graph_square/'):
    os.makedirs('./save/generated_2d_graph_square/')
if not os.path.exists('./save/generated_crystal_for_check/'):
    os.makedirs('./save/generated_crystal_for_check/')

filenames=os.listdir('./calculation/generated_crystal_for_check/')

for filename in filenames:
    command='cp ./calculation/generated_crystal_for_check/'+filename+' ./save/generated_crystal_for_check/'
    os.system(command)
    command='cp ./calculation/generated_2d_graph_square/'+filename[:-4]+'.npy'+' ./save/generated_2d_graph_square/'
    os.system(command)

command='rm -r ./calculation/generated_*/;'
os.system(command)


print（'the generation process has finished'）