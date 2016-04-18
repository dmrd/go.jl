wget https://julialang.s3.amazonaws.com/bin/linux/x64/0.4/julia-0.4.5-linux-x86_64.tar.gz
tar xfz julia-0.4.5-linux-x86_64.tar.gz
cd julia-2ac304dfba

# Do some stuff with conda

#Pkg.add("theano")
#Pkg.add("cudatoolkit")

# Install CUDA

sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb 
sudo apt-get update
sudo apt-get install cuda


# Setup CUDA paths

echo "export CUDA_HOME=/usr/local/cuda-7.5" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64"  >> ~/.bashrc
echo "PATH=${CUDA_HOME}/bin:${PATH}"  >> ~/.bashrc
echo "export PATH" >> ~/.bashrc
