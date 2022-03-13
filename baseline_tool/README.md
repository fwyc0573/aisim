# Baseline tool

Baseline tool provides a Reproducible, Easy-to-use, Reusable tool to run some commonly used CNN and RNN models as a baseline.

It can ouput the time-use and loss by step, metadata of tfprof and tensorflow timeline. you can customize the session times and step times, and you can decide if the horovod and multi-GPUs turn on or off.

## Code Structure

### File Structure

- `baseline_tool` : project to do real device profiling
  - `profile.py`: the command line tool to use directly
  - `README.md`: the README document of this project
  - `Dockerfile`: the Docker to run this project
  - `tests`: the unitest programs
  - `tf_profile`: the source code of tensorflow profiling
    - `tf_profile.py` : the python API used by profile.py to do profiling
    - `models` : the module to call different models
      - `tf_model.py` : the python API used by tf_profile.py to call models
      - `RNN` : the source code of RNN models
      - `CNN` : the source code of CNN models

### Program Structure

![program_structure](README.assets/profiling_program_structure.png
)

## Installation

### Install by Dockerfile

Using Docker to get a quickstart.

First, build the docker image using the DockerFile in the root directory of the project 

```bash
$ sudo docker build -t <image_name>:<tag> .
```

Then, run the docker image

```bash
$ sudo nvidia-docker run -it <image_name>:<tag>
```

If you want to get multiple workers' performance with RDMA, run the docker image with these options in every machine

```bash
$ sudo nvidia-docker run -it --network=host --privileged  -v /root/.ssh:/root/.ssh --cap-add=IPC_LOCK --device=/dev/infiniband baseline
```

Only one main machine run the python file, which must be able to successfully connect to other machines using ssh

- Use sshd to listen to a specifc port in the container that will be connected

  ```bash
  $ /usr/sbin/sshd -p <available_port>
  ```

- Configure your ~/.ssh/config file to assign custom host names and ports for each container

  ```bash
  $ vim /root/.ssh/config
  ```

  ```bash
  Host host1
    HostName 192.168.1.10
    Port 1234
  ```

- Add public key to config the ssh authentication in each container

   ```
   $ vim /root/.ssh/authorized_keys
   ```
  
- Check ssh, the main worker's docker should login the other workers' docker without password

Check the connectivity of infiniband

- perftest is prepared in the docker, you can refer to https://github.com/linux-rdma/perftest to see more
  
   ```
   $ cd perftest
   ```

- host1 listen on the port

   ```
   $ ./ib_write_bw -a -d mlx5_1 -p 10008
   ```

- host2 connect to the port of host1:

   ```
   $ ./ib_write_bw -a -d mlx5_1 -p 10008 host1
   ```

Check nccl

 ```
 $ cd /nccl-tests/build
   
 $ mpirun -np 2 -H 172.23.232.139,172.23.232.166 \
          -bind-to none -map-by slot \
       -x LD_LIBRARY_PATH -x PATH \
       -x NCCL_DEBUG=INFO \
       -x NCCL_SOCKET_IFNAME=^lo,docker0 \
       -mca pml ob1 -mca btl ^openib \
       -mca btl_tcp_if_exclude lo,docker0 \
       -x HOROVOD_MPI_THREADS_DISABLE=1 \
   -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_1 \
   -x NCCL_IB_GID_INDEX=0 -x NCCL_IB_CUDA_SUPPORT=1 \
   	--allow-run-as-root \
   ./all_reduce_perf -b 8 -e 1024M -f 2 -g 4
   ```

### Install on Native Machine

It's recommended to use conda to configure the environment.

Install miniconda first

```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

Create you own conda environment, install Python3.7 and OpenMPI which is the base of horovod

```bash
$ conda create --name <conda_name> python=3.7 openmpi
$ conda activate <conda_name>
```

Install tensorflow

```bash
$ python -m pip install tensorflow==1.15
```
Install nccl and horovod

```bash
$ git clone https://github.com/NVIDIA/nccl.git
$ cd nccl
$ make -j20 src.build CUDA_HOME=/usr/local/<cuda-version>
# cuda-10.0 can be run successfully
$ HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=<nccl_path>/build HOROVOD_CUDA_HOME=/usr/local/<cuda-vesrsion>/ python -m pip install --no-cache-dir horovod
# cmake is required for Horovod installation. Please use 'sudo apt install cmake' for ubuntu.
```

Verify the conda environment

```bash
$ horovodrun --check-build
```

## Quickstart

### To get **single GPU**'s performance

To get the list of models available

```bash
$ python profile.py --list
```

To get all models’ time of steps *(-t)*, set output folder *(-o)* to ./output/  (default 10 sessions 600 steps)

```bash
$ mkdir output
$ python profile.py –o output -t
```

To get resnet50’s *(-m)* tfprof *(-p)* and timeline *(--time-line)* metadata, use 1 session and 20 step

```bash
python profile.py –m resnet50 –p –-time-line --session 1 --step 20
```

To get all models' computation graphs *(-g)* and set output folder *(-o)* to ./graphs/

```bash
$ mkdir graphs
$ python profile.py -o graphs --graph 
```

### To get **multiple GPUs**' horovod performance

Two way to use horovod to run the model:

- let the python script to setup horovodrun:
  
  ```bash
  $ python profile.py --horovod -n 4
  ```

- setup horovodrun manually

  ```bash
  $ horovodrun -np 4 -H localhost:4 python profile.py 
  ```

### To get **multiple workers**' horovod performance

Use mpirun directly with each worker's ip

```bash
$ mpirun -np 8 -H host1:4,host2:4 \
       -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
-x NCCL_DEBUG=INFO \
-x NCCL_DEBUG_SUBSYS=all \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_1 \
-x NCCL_IB_GID_INDEX=0 -x NCCL_IB_CUDA_SUPPORT=1 \
	--allow-run-as-root \
	python profile.py -m alexnet 
```

## Usage

use a cmd-line tool `python profile.py` to do the profiling

- `-h` or `--help`: get the hlep information

- `--list`: list out all the models available

- `--session <session_num>`
  - set the *session_num per model* of the profiling
  - default to 10

- `--step <step_num>`
  - set the *step_num per session*
  - default to 600

- `-m <model name1 name2>`
  - select which model to be run
  - default all models

- `--horovod`
  - open horovod mode by force
  - horovod will auto open if use horovodrun with GPUs > 1

- `-n`
  - set the gpu_num by force

- `-o <folder_path>`
  - set the output folder of the *.csv* and other output
  - (default to *-o ./*)

There are 5 options to use with -o:

- `-g` or `--graph`
  - let the program to output the computing graphs of model
  - will be saved in `-o`'s folder
  - **notice:** even works when *--session 0*, **no** influence to performance

- `-t` or `--time`
  - let the program to output time of steps
  - *.csv* default to *model_GPUs_horovod.csv*
  - will be saved in `-o`'s folder

- `-l` or `--loss`
  - let the program to output loss of steps
  - will be saved in `-o`'s folder
  - **notice:** this process may have a little influence to performance

- `-p` or `--tfprof`
  - let the program to output the *tfprof* metadata
  - will be saved in `-o`'s folder
  - **notice:** this process may have influence to performance

- `--time-line`
  - let the program to output the timeline metadata
  - will be saved in `-o`'s folder
  - **notice:** this process may have influence to performance


## Models Information

- `VGG16`
  - **default batchsize**: 64
  - **dataset**: imagenet (all one) [224,224,3]
- `resnet50`
  - **default batchsize**: 32
  - **dataset**: imagenet (all one) [224,224,3]
- `inception3`
  - **default batchsize**: 32
  - **dataset**: imagenet (all one) [299,299,3]
- `bert`
  - **default batchsize**: 32
  - **dataset**: Glue (random) [32,7,30522]
- `bert_large`
  - **default batchsize**: 32
  - **dataset**: Glue (random) [32,7,30522]
- `nasnet`
  - **default batchsize**: 5
  - **dataset**: imagenet (random) [224,224,3]
- `alexnet`
  - **default batchsize**: 128
  - **dataset**: cifar10 (all one) [32,32,3]
- `lstm`
  - **default batchsize**: 16
- `seq2seq`
  - **default batchsize**: 16
- `deepspeech`
  - **default batchsize**: 16

