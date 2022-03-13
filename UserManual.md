# User Manual for AISim Benchmark

Use the ai_simulator/simulator_benchmark in the AISimGo to simulate the model running performance, measure the simulation's accuracy and op/node coverage.

## Setup Environment

Currently, we use tensorflow1.15 and python 3.6/3.7 as our basic environment configuration.

To successfully run AISim, please make sure you have the total codes of SuperScalar and `$AISimGo/ai_simulator` either in your docker or local machine.

**Build Docker**

AISimGo is an instance ultilizing AISim's capability to illustrate the procedure of simulation of an AI system.

You can use `$AISimGo/ai_simulator/Dockerfile` to build your docker image.
```
sudo docker build -t ai_simulator .
```
   
## Change the input resource

To simulate machine performance, there are some input resources need to be changed.

You should change the resources of different GPU type(eg, MI100,V100) and GPU number(eg, 1GPU,2GPU,4GPU,8GPU) including database, graphs and resource_pool.yaml according to your requirements.

Here we take MI100 single gpu for example.

- Use the **database** of specific hardware

    There are 2 choices, one is use a single total database of specific GPU type, or you can use multiple databases specific to the model and GPU.

  - databases specific to the model and GPU
  
    Please copy the contents under `$AISimGo/resource/MI100/1GPU/db/` to `$AISimGo/ai_simulator/simulator_benchmark/data/database/`.
  - database specific to GPU type
  
    Please copy the `db/MI100_db.json` to `ai_simulator/simulator_benchmark/data/database` and replace 
`'./data/database/%s_db.json' % model` to `./data/database/MI100_db.json` in `ai_simulator/simulator_benchmark/benchmark_tools.py`.

- Get the **graphs** needed by simulator

   Please copy the contents under `%AISimGo/resource/MI100/1GPU/graphs/` to `%AISimGo/ai_simulator/simulator_benchmark/data/graphs/`.

- Replace the **resource configuration**
  
  Please replace the  `%AISimGo/ai_simulator/simulator_benchmark/data/resource_pool.yaml` with `%AISimGo/resource/MI100/1GPU/resource_pool.yaml`.


## How to Run

```
cd simulator_benchmark
```

**Quick Start:**

Run all models:

```
python simulator_benchmark.py
```

**Params:**

-m {model [model1…]}: 

```
python simulator_benchmark.py -m alexnet vgg16
```

--skip-coverage: Skip the database coverage check

--skip-accuracy: Skip the simulator accuracy benchmark

**Sample Output**:

```
$ python3 simulator_benchmark.py -m alexnet vgg16

Coverage of database using single GPU graph
model      OP coverage         node coverage       uncoveraged OP(count)
alexnet    90/96=93.8%         17/21=81.0%         Conv2D(2) Mul(1) Conv2DBackpropInput(1) Conv2DBackpropFilter(2) 
vgg16      216/255=84.7%       15/19=78.9%         Conv2D(13) Mul(1) Conv2DBackpropInput(12) Conv2DBackpropFilter(13) 
Average op coverage: 89.2


Coverage of database using multiple GPUs graph
model      OP coverage         node coverage       uncoveraged OP(count)
alexnet    180/216=83.3%       17/24=70.8%         Const(2) Conv2D(4) Mul(2) _HostRecv(2) ...
vgg16      432/576=75.0%       15/21=71.4%         Const(2) Conv2D(26) Mul(2) Switch(64) ...
Average op coverage: 79.2


Timeuse(ms) and Accuracy(%) of simulator:
Environment         alexnet             vgg16               average_loss        
1 Host X 1 V100     4.4/5.5=79.1%       53.1/270.4=19.6%    50.6%
1 Host X 2 V100     4.4/6.7=65.3%       67.9/302.7=22.4%    56.1%
1 Host X 4 V100     4.4/7.0=62.2%       88.5/318.3=27.8%    55.0%
2 Host X 4 V100     4.4/7.0=62.7%       98.8/334.1=29.6%    53.9%
average_loss        32.7%               75.1%               53.9%               
Total average absolute loss: 53.9%
```

Note: the value before `/` in `accuracy` is the simulation result of each model and the value after `/` should be modified to real baseline results in `ai_simulator/simulator_benchmark/data/baseline.json`.

## Notes

Available Models:

- `vgg16`
- `resnet50`
- `inception3`
- `alexnet`
- `lstm`
- `seq2seq`
- `deepspeech`
- `bert`
- `bert_large`




