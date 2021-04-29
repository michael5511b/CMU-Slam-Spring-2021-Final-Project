# Stuff that you might need
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev
sudo apt-get install libcgal-qt5-dev
sudo apt-get install libatlas-base-dev libsuitesparse-dev

# ceres
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
sudo make install

# Colmap
git clone https://github.com/colmap/colmap
cd colmap
# Removing git from colmap

rm -R .github 
rm .gitignore
rm .gitmodules
git checkout dev
mkdir build
cd build
cmake ..
make -j
sudo make install

# Demo dataset download
cd ..
pip install gdown
gdown https://drive.google.com/uc?id=1ZjuNPGIDr89EXJD0Z7TeV2sAymOa69ps
tar -xf gerrard-hall.tar.gz