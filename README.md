<img src="tensorsocket_logo_medium.png" align="left" width="200px"/>

# tensorsocket

Share PyTorch tensors over ZMQ sockets
<br clear="left"/>

## Installation

tensorsocket can either be installed as a module, or installed from pypi.

**From source**

From the root of the tensorsocket directory, install it with pip:

```shell
$ pip install .
```

**From PyPi**

```shell
$ pip install tensorsocket
```

## Usage

tensorsocket works by exposing batches of data, represented as PyTorch tensors, on sockets that training processes can access. This allows for minimizing redundancy of training data during collocated tasks such as hyper-parameter tuning. Training with tensorsocket builds on the concept of a producer-consumer relationship, where the following example code shows how the producer wraps around an arbitrary data loader object. As with nested epoch-batch loops, one can iterate over the producer in the same manner as iterating over a data loader.

The use of tensorsocket relies on a `TensorProducer` and `TensorConsumer`. 

Using the `TensorProducer` requires next to no additional implementation, apart from the original data loader. 

```python
# producer.py
data_loader = DataLoader(dataset)

producer = TensorProducer(data_loader)

for _ in range(epochs):
        for _ in producer:
            pass
producer.join()
```

It is straightforward to modify a training script to fetch batches of data from the shared loader, rather than using the process-specific data loader, which is created for each collocated training job.

```python
# consumer.py (or train.py)
if not use_shared_loader:
    data_loader = create_loader(...)
else:
    data_loader = TensorConsumer()

...

for batch_idx, (input, target) in enumerate(data_loader):
    output = model(input)
    ...

```

There is a range of arguments that can be supplied to the `TensorProducer` and `TensorConsumer`, altering their behaviour to your specific needs. Examples include more/less aggressive buffering, flexible batch sizing, and larger rubberbanding buffers.
