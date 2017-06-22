cd $TRAVIS_BUILD_DIR

if [[ ${TASK} == "pylint" ]]; then
    pip install pep8
    pep8 --ignore=E501 --exclude=./compute,./docs . || exit -1
    exit 0
fi

conda install --yes numpy scipy scikit-learn pandas matplotlib
pip install pytest

if [[ $TRAVIS_OS_NAME == "linux" ]]; then 
    conda install --yes atlas
    conda install --yes -c conda-forge boost=1.63.0
fi

mkdir build && cd build

if [[ ${TASK} == "regular" ]]; then
    cmake ..
elif [[ ${TASK} == "mpi" ]]; then
    cmake -DUSE_MPI=ON ..
elif [[ ${TASK} == "gpu" ]]; then
    cmake -DUSE_GPU=ON -DBOOST_ROOT="$HOME/miniconda/" -DOpenCL_INCLUDE_DIR=$AMDAPPSDK/include/ ..
    sed -i 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' ../include/LightGBM/config.h
fi

make _lightgbm || exit -1

cd $TRAVIS_BUILD_DIR/python-package && python setup.py install --precompile || exit -1
cd $TRAVIS_BUILD_DIR && pytest . || exit -1
