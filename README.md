# CCDCGAN-for-single-system
The purpose of this code is to design new crystal structures of Bi-Se system with low formation energy. This code is related to the work by T. Long, et al. "Constrained crystals deep convolutional generative adversarial network for the inverse design of crystal structures." npj Computational Materials 7.1 (2021): 1-7. Please feel free to contact the conresponding author Prof. Hongbin Zhang (hzhang@tmm.tu-darmstadt.de) or Teng Long (tenglong@tmm.tu-darmstadt.de) for discussions.

## Preparation of the environment
We use the following environment to run this code, it has been tested by our colleborators as well. Please make sure that you have the same environment as us before running this code, please check them carefully if you have unexpected errors.

```vim
Name                    Version                   Build       Channel
_libgcc_mutex             0.1                        main  
_tflow_select             2.1.0                       gpu  
absl-py                   0.10.0                   py37_0  
ase                       3.20.1                     py_0    conda-forge
astor                     0.8.1                    py37_0  
blas                      1.0                         mkl  
c-ares                    1.16.1               h7b6447c_0  
ca-certificates           2020.10.14                    0  
certifi                   2020.11.8        py37h06a4308_0  
click                     7.1.2              pyh9f0ad1d_0    conda-forge
cudatoolkit               10.1.243             h6bb024c_0  
cudnn                     7.6.5                cuda10.1_0  
cupti                     10.1.168                      0  
cycler                    0.10.0                     py_2    conda-forge
flask                     1.1.2              pyh9f0ad1d_0    conda-forge
freetype                  2.10.4               h7ca028e_0    conda-forge
gast                      0.4.0                      py_0  
google-pasta              0.2.0                      py_0  
grpcio                    1.31.0           py37hf8bcb03_0  
h5py                      2.10.0           py37hd6299e0_1  
hdf5                      1.10.6               hb1b8bf9_0  
importlib-metadata        2.0.0                      py_1  
intel-openmp              2020.2                      254  
itsdangerous              1.1.0                      py_0    conda-forge
jinja2                    2.11.2             pyh9f0ad1d_0    conda-forge
joblib                    0.17.0                     py_0  
jpeg                      9d                   h36c2ea0_0    conda-forge
keras-applications        1.0.8                      py_1  
keras-preprocessing       1.1.0                      py_1  
kiwisolver                1.3.1            py37hc928c03_0    conda-forge
lcms2                     2.11                 hcbb858e_1    conda-forge
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libpng                    1.6.37               h21135ba_2    conda-forge
libprotobuf               3.13.0.1             hd408876_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h4f3a223_6    conda-forge
libwebp-base              1.1.0                h36c2ea0_3    conda-forge
lz4-c                     1.9.2                he1b5a44_3    conda-forge
markdown                  3.3.2                    py37_0  
markupsafe                1.1.1            py37hb5d75c8_2    conda-forge
matplotlib-base           3.3.3            py37h4f6019d_0    conda-forge
mkl                       2020.2                      256  
mkl-service               2.3.0            py37he904b0f_0  
mkl_fft                   1.2.0            py37h23d657b_0  
mkl_random                1.1.1            py37h0573a6f_0  
ncurses                   6.2                  he6710b0_1  
numpy                     1.19.1           py37hbc911f0_0  
numpy-base                1.19.1           py37hfa32c7d_0  
olefile                   0.46               pyh9f0ad1d_1    conda-forge
openssl                   1.1.1h               h7b6447c_0  
pillow                    8.0.1            py37h63a5d19_0    conda-forge
pip                       20.2.4                   py37_0  
protobuf                  3.13.0.1         py37he6710b0_1  
pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
python                    3.7.9                h7579374_0  
python-dateutil           2.8.1                      py_0    conda-forge
python_abi                3.7                     1_cp37m    conda-forge
readline                  8.0                  h7b6447c_0  
scipy                     1.5.2            py37h0b6359f_0  
setuptools                50.3.0           py37hb0f4dca_1  
six                       1.15.0                     py_0  
sqlite                    3.33.0               h62c20be_0  
tensorboard               1.14.0           py37hf484d3e_0  
tensorflow                1.14.0          gpu_py37h74c33d7_0  
tensorflow-base           1.14.0          gpu_py37he45bfe2_0  
tensorflow-estimator      1.14.0                     py_0  
tensorflow-gpu            1.14.0               h0d30ee6_0  
termcolor                 1.1.0                    py37_1  
tk                        8.6.10               hbc83047_0  
tornado                   6.1              py37h4abf009_0    conda-forge
werkzeug                  1.0.1                      py_0  
wheel                     0.35.1                     py_0  
wrapt                     1.12.1           py37h7b6447c_1  
xz                        5.2.5                h7b6447c_0  
zipp                      3.3.1                      py_0  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.5                h6597ccf_2    conda-forge
```

## Decompress the "database.rar" in the "main" folder
Please make sure that the "database" folder is in the "main" folder, which consists of two folders, i.e., "geometries" and "properties".

## Instructions of running
1.Train the model first by type "python train_GAN.py" in the terminal. 

2.The training process is only sucessful after seeing "the training process has finished" in the terminal.

3.Generate new structures by type "python generate_new_structure.py" in the terminal.

4.The generation process is only sucessful after seeing "the generation process has finished" in the terminal.

## Notes
1.Please note that all generated strutures will be overwrite the previous one.

2.Please also note that at least 100GB hard disk is required during the training process.
