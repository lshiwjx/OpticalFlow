# Calculate OpticalFlow with (multi)CPU/GPU
/cflow is the C++ realization of the algorithm. \
/pyflow is a python warper\
Directly decoding from video can save much IO time. 

## Requirement
Opencv >= 3.0 with GPU Support and Contrib installed

## Usage
mkdir build\
cd build\
cmake ..\
make\

Their will be three executable files in bin/

## Comparison
Original images: \
![alt text](data/00001.jpg)
![alt text](data/00005.jpg)

Results of different methods: \
![alt text](resource/compare_flow_methods.png)

Results using gpu/cpu mehtods: \
![alt text](resource/compare_gpu_cpu.png)