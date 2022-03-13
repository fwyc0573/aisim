# AI Simulator Benchmark

This is a benchmark tool to measure AI Simulator's accuracy and database's coverage.

## Installation

- Add SuperScaler package

  ```
  export PYTHONPATH=$PYTHONPATH:<Path to SuperScaler>/src
  ```


## How to Use

**Quick Start:**

Run all 8 models:

```
python3 simulator_benchmark.py
```

**Other Options:**

Run seleceted models:

```
python3 simulator_benchmark.py -m alexnet inception3 vgg16
```

Skip the database coverage check:

```
python3 simulator_benchmark.py --skip-coverage
```

Skip the simulator accuracy benchmark:

```
python3 simulator_benchmark.py --skip-accuracy
```

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

## Notes

Available Models:

- `vgg16`
- `resnet50`
- `inception3`
- `nasnet` **No Multi-GPU graph, cannot run horovod model currently**
- `alexnet`
- `lstm`
- `seq2seq`
- `deepspeech`



