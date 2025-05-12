import logging
import threading
import time
import zmq

import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from torch import Tensor, cuda, cat
from torch.nn.utils.rnn import pad_sequence
from tornado import ioloop
from typing import Any, Iterator, Optional
from zmq.eventloop import zmqstream

from .heartbeat import HeartBeater
from .payload import TensorPayload

logger = logging.getLogger("tensorsocket")
# logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://*"


@dataclass
class ConsumerProgress:
    """Tracks progress of individual consumers through dataset.

    Maintains ordered queues for batch IDs and payloads to ensure sequential processing
    and proper synchronization between producer and consumer.

    Attributes:
        id_queue: Queue storing sequential batch IDs
        payload_queue: Queue storing corresponding tensor payloads
    """

    id_queue: deque[int]
    payload_queue: deque[dict]  # Deprecated
    loader_batch_size: int
    batch_size: int

    def add_batch(self, id: int, payload: dict) -> None:
        """Add a new batch to tracking queues.

        Args:
            id: Sequential batch identifier
            payload: Associated tensor data and metadata

        Raises:
            AssertionError: If batch ID is not sequential
        """
        # assert id == self.batch_max
        self.id_queue.append(id)
        # self.payload_queue.append(payload)

    def remove_batch(self, id: int) -> None:
        """Remove the leftmost ID/payload pair. ID must match arg0.

        Args:
            id: Sequential batch identifier

        Raises:
            AssertionError: If batch ID does not match the leftmost ID
        """
        # assert id == self.batch_count
        self.id_queue.popleft()
        # self.payload_queue.popleft()

    def reset(self):
        """Clear all stored IDs and payloads."""
        self.id_queue.clear()
        self.payload_queue.clear()

    # @property
    # def batch_count(self) -> int:
    #     """Get the current batch count.

    #     Returns:
    #         The leftmost batch ID in the queue.
    #     """
    #     return self.id_queue[0] * self.batch_size // self.loader_batch_size

    @property
    def batch_max(self) -> int:
        """Get the maximum batch ID.

        Returns:
            The rightmost batch ID in the queue plus one, or 0 if the queue is empty.
        """
        if self.loader_batch_size == 0:
            return 0

        return (
            (self.id_queue[-1] + 1 if self.id_queue else 0)
            * self.batch_size
            // self.loader_batch_size
        )


def process_tensor(tensor: Any) -> TensorPayload:
    """Process single tensor for transmission.

    Args:
        tensor: PyTorch tensor to process

    Returns:
        Processed tensor wrapped in TensorPayload
    """
    if isinstance(tensor, Tensor):
        tensor = TensorPayload(tensor)
    return tensor


def data_to_cuda(data: tuple) -> tuple:  # TODO: also dict
    if isinstance(data, dict):
        return {
            k: (v.to(device="cuda") if isinstance(v, Tensor) else v)
            for k, v in data.items()
        }
    return tuple(
        (element.to(device="cuda") for element in data if isinstance(element, Tensor))
    )


def pack(data) -> tuple:  # TODO: also dict
    """Pack multiple tensors for transmission.

    Args:
        data: Tuple of tensors to process

    Returns:
        Tuple of processed tensors
    """
    if isinstance(data, dict):
        return {k: process_tensor(v) for k, v in data.items()}
    return tuple((process_tensor(t) for t in data))


def slice(data, a: int, b: int) -> tuple:  # TODO: also dict
    """Slice multiple tensors"""
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = data[key][a:b]
        return new_data
    return tuple((element[a:b] for element in data))


def collate(batches: list) -> tuple:  # TODO: also dict
    """Collate multiple tensors"""
    if isinstance(batches[0], dict):
        new_batch = {}
        max_length = -1
        for key in batches[0].keys():
            new_batch[key] = pad_sequence(
                [line for batch in batches for line in batch[key]],
                batch_first=True,
                padding_value=0,
            )

            if (i:=new_batch[key].shape[-1]) > max_length:
                max_length = i
        
        for key in batches[0].keys():
            if (new_length:=max_length - new_batch[key].shape[-1]) > 0:
                if key == "labels":
                    new_batch[key] = F.pad(new_batch[key], (0, new_length), value=-100)
                else:
                    new_batch[key] = F.pad(new_batch[key], (0, new_length), value=0)
        

        return new_batch
    return tuple((cat([batch[i] for batch in batches]) for i in range(len(batches[0]))))


class TensorPool:
    """A circular buffer for tensor data management.
    PyTorch's shared memory operation leaks memory, which is why we need to use a pool.
    This class implements a fixed-size circular buffer (pool) for managing tensor data,
    with support for CUDA devices if available. It handles automatic overwriting of old
    data when the pool is full.
    """

    def __init__(self, max_size: int = 10) -> None:
        self.pool = [None] * max_size
        self.max_size = max_size
        self.index = 0

    def _overwrite_data(self, destination: tuple, source: tuple) -> tuple:
        """Overwrite data from source to destination tensors.
        This function copies data from source tensors to destination tensors in-place.
        Args:
            destination (tuple): Tuple containing destination tensors to be overwritten
            source (tuple): Tuple containing source tensors to copy from
        Returns:
            tuple: The destination tuple with updated tensor values
        """
        # destination = source  # TODO: fix
        # return destination

        if isinstance(source, dict):
            for key in source.keys():
                if isinstance(source[key], Tensor):
                    destination[key].copy_(source[key])
        else:
            for dest, src in zip(destination, source):
                if isinstance(src, Tensor):
                    dest.copy_(src)
        return destination

    def assign(self, data: tuple) -> None:
        """
        Assigns data to the next available slot in the circular buffer.
        Args:
            data (tuple): Data to be assigned to buffer.
        Returns:
            The data element that was just assigned to the buffer.
        """

        self.index = (self.index + 1) % self.max_size

        if self.pool[self.index] is not None:
            self._overwrite_data(self.pool[self.index], data)
        else:
            if cuda.is_available():
                data = data_to_cuda(data)
            self.pool[self.index] = data

        return self.pool[self.index]


class TensorProducer:
    """Distributes tensor batches to multiple training processes over ZMQ.

    Handles:
    - Dataset iteration and batch distribution
    - Consumer health monitoring via heartbeat
    - Acknowledgement tracking
    - Rubberbanding for consumer synchronization
    - Automatic cleanup of disconnected consumers
    """

    def __init__(
        self,
        data_loader: Iterator,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: tuple[int, int] = (4444, 4445),
        rubber_band_pct: float = 0.005,
        pack_fn: callable = pack,
        consumer_max_buffer_size: int = 10,
        producer_batch_size: int = 8,  # TODO: divide
        max_tensor_length: int = 150,
    ) -> None:
        """Initialize producer with configuration.

        Args:
            data_loader: Source of tensor batches
            port: Main data transmission port
            ack_port: Consumer acknowledgement port
            heart_ports: (Pub, Sub) ports for heartbeat monitoring
            rubber_band_pct: Max allowed consumer lag as % of dataset
            pack_fn: Function to prepare tensors for transmission
            consumer_max_buffer_size: Max queued batches per consumer
        """
        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.data_loader = data_loader
        self.data_loader_len = len(self.data_loader)
        try:
            self.data_loader_iter = iter(self.data_loader)
        except TypeError:
            raise TypeError("TensorSocket: DataLoader is already iterable")

        self.pack_fn = pack_fn
        self.producer_batch_size = producer_batch_size
        self.loader_batch_size = 0
        self.pool = TensorPool(consumer_max_buffer_size)

        self.index = 0
        self.context = zmq.Context()
        self.consumers = {}
        self.consumer_max_buffer_size = consumer_max_buffer_size
        self.send_len = False

        # Send batches via
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"{LOCALHOST}:{self.port}")

        # Ack
        self.ack_socket = self.context.socket(zmq.SUB)
        self.ack_socket.bind(f"{LOCALHOST}:{self.ack_port}")
        self.ack_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Heartbeat monitor
        self.hb = None
        self.heartbeat_thread = threading.Thread(
            target=self._start_heartbeat, args=(), daemon=True
        )
        self.heartbeat_thread.start()

        while self.hb is None:
            time.sleep(0.2)

        self._heartbeat_monitor_alive = True
        self.heartbeat_monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, args=(), daemon=True
        )
        self.heartbeat_monitor_thread.start()

        # Dataset logic
        self.dataset_is_reset = True
        self.epoch = 0
        self.max_tensor_length = max_tensor_length

        # Rubberbanding
        self.rb_buffer = list()
        self.rb_max_len = rubber_band_pct * self.data_loader_len
        self.rb_running = False

        self.active_payloads = []

    def _start_heartbeat(self) -> None:
        """Initialize and start heartbeat monitoring system.

        Sets up ZMQ pub/sub socket pair for tracking consumer connections
        and initializes event loop for async heartbeat processing.
        """
        self.loop = ioloop.IOLoop()
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.bind(f"{LOCALHOST}:{self.heart_ports[0]}")
        sub = context.socket(zmq.SUB)
        sub.bind(f"{LOCALHOST}:{self.heart_ports[1]}")
        sub.subscribe(b"")

        outstream = zmqstream.ZMQStream(pub, self.loop)
        instream = zmqstream.ZMQStream(sub, self.loop)

        self.hb = HeartBeater(self.loop, outstream, instream)
        self.loop.start()

    def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat in background thread.

        Runs continuously until producer shutdown, checking consumer health.
        """
        while True:
            if not self._heartbeat_monitor_alive:
                break
            time.sleep(1)

    def join(self) -> None:
        """Clean shutdown of producer threads and event loops.

        Stops heartbeat monitoring and joins all background threads.
        """
        self.loop.stop()
        self.heartbeat_thread.join()
        self._heartbeat_monitor_alive = False
        self.heartbeat_monitor_thread.join()

    def __iter__(self) -> Iterator:
        """Return an iterator for the data loader.
        Waits for all consumers to be ready before starting.

        Returns:
            An iterator for the data loader.
        """
        # Block until all consumers are ready
        while True:
            for k, v in self.consumers.items():
                if self.hb.heart_progress[k] == 0:
                    v.reset()
                else:
                    break
            else:
                break

        return self

    def __next__(self) -> tuple:
        """Get the next sample from the data loader.

        Returns:
            The next sample from the data loader.
        """
        return self.get_sample()

    def get_sample(self) -> Optional[tuple]:
        """Core batch distribution method.

        Handles:
        1. Dead consumer cleanup
        2. Epoch boundaries and dataset reset
        3. Consumer synchronization via rubberbanding
        4. Batch transmission and acknowledgement processing

        Returns:
            Tensor batch or None if no active consumers

        Raises:
            StopIteration: At end of epoch
        """
        # clean up dead consumers
        for consumer in list(self.consumers.keys()):
            if consumer not in self.hb.consumers:
                self.consumers.pop(consumer)

        if self.index >= self.data_loader_len:
            self._reset_producer()
            return

        # idle when no consumers attached
        elif not len(self.hb.consumers):
            logger.info("No consumers, waiting ...")
            time.sleep(0.5)
            return

        current_batch_index = self.index

        try:
            expected = [str(x) for x in self.hb.consumers]

            for consumer in expected:
                if str(consumer) not in self.consumers:
                    self.consumers[str(consumer)] = ConsumerProgress(
                        deque(maxlen=self.consumer_max_buffer_size + 1),
                        deque(maxlen=self.consumer_max_buffer_size + 1),
                        self.loader_batch_size,
                        self.hb.heart_batch_size[str(consumer)],
                    )
                    self.send_len = True

            # Can only send len if loader_batch_size is known
            if self.send_len and self.loader_batch_size:
                self.send_len = False
                self._send_consumer_len()

            if self.rb_buffer:
                try:
                    if (
                        min_batch := min(
                            [
                                x.batch_max
                                for x in self.consumers.values()
                                if x.batch_max
                                >= self.index
                                - len(
                                    self.rb_buffer
                                )  # Ignore consumers that are too far behind
                            ]
                        )
                    ) not in [x[0] for x in self.rb_buffer]:
                        current_batch_index, buffer_index = -1, -1
                    else:
                        buffer_index = [x[0] for x in self.rb_buffer].index(min_batch)
                        current_batch_index, _ = self.rb_buffer[buffer_index]
                except ValueError as e:
                    # No valid consumers, reset
                    raise StopIteration

            else:
                current_batch_index, buffer_index = 0, 0

            batch_length = len(self.rb_buffer[buffer_index:])

            if batch_length < self.producer_batch_size:
                # add CPU tensors to rubberband buffer
                n = next(self.data_loader_iter)
                if isinstance(n, dict):
                    for key in n.keys():
                        if isinstance(n[key], Tensor):
                            n[key] = F.pad(
                                n[key], (0, self.max_tensor_length - n[key].shape[-1]), value=0
                            )
                self.rb_buffer.append((self.index, n))

                # if loader batch size not yet determined, set it
                if self.loader_batch_size == 0:
                    self.loader_batch_size = len(self.rb_buffer[-1][1][0])

                    for consumer in self.consumers:
                        self.consumers[consumer].loader_batch_size = (
                            self.loader_batch_size
                        )

                # if buffer full, pop from end
                if len(self.rb_buffer) > self.rb_max_len:
                    _ = self.rb_buffer.pop(0)

                self.index += 1

                return

            expected = [x for x in expected]
            payload = self._broadcast(self.epoch, current_batch_index, buffer_index)
            self._handle_acks(expected, payload)

        except StopIteration:
            self._reset_producer()
            return

    def _reset_producer(self):
        self.data_loader_iter = iter(self.data_loader)
        self.index = 0
        self.epoch += 1
        self.rb_buffer = []
        self.socket.send_pyobj({"stop_iteration": self.epoch})
        raise StopIteration

    def _broadcast(
        self, current_epoch: int, current_batch_index: int, buffer_index: int
    ) -> dict:
        """Broadcast tensor batch to all connected consumers.

        Args:
            current_epoch: Training epoch number
            current_batch_index: Current batch index in epoch
            data: Tensor batch to transmit

        Returns:
            Dict containing packed tensors and metadata

        Logs progress every 100 batches.
        """

        data = collate(
            [
                x[1]
                for x in self.rb_buffer[
                    buffer_index : buffer_index + self.producer_batch_size
                ]
            ]
        )

        data = self.pool.assign(data)

        payload = {}
        for consumer, bs, bmax in (
            (k, v.batch_size, v.batch_max) for k, v in self.consumers.items()
        ):

            # If the consumer is too far behind, skip
            if bmax < self.index - len(self.rb_buffer):
                continue

            messages = []
            try:
                lens = len(data["labels"])  # TODO: fix
            except TypeError as e:
                lens = 0

            for i, offset in enumerate(
                range(
                    (bmax - current_batch_index) * self.loader_batch_size,
                    lens,  # TODO: fix
                    bs,
                )
            ):
                if offset + bs > lens:  # TODO: fix
                    break

                messages.append(
                    dict(
                        data=(n := self.pack_fn(slice(data, offset, offset + bs))),
                        current_epoch=current_epoch,
                        current_batch_index=bmax * self.loader_batch_size // bs + i,
                    )
                )

            payload[consumer[2:-1]] = messages

        if current_batch_index % 100 == 0:
            logger.info(
                f"current_batch_index {current_batch_index}, "
                f"buffer size: {len(self.rb_buffer)}"
            )

        self.socket.send_pyobj(payload)
        return payload

    def _handle_acks(self, expected: list, payload: dict) -> None:
        """Process acknowledgements from consumers.

        Args:
            expected: List of consumer IDs to wait for
            current_batch_index: Index of batch being acknowledged
            payload: Transmitted data payload

        Times out after 10 seconds if consumers unresponsive.
        Maintains synchronization by tracking accepted batches.
        """
        while True:
            # received all Acks, can go to next batch
            if not len(expected):
                return

            if self.ack_socket.poll(10000, zmq.POLLIN):
                (
                    consumer_index,
                    batch_max,
                    accepted,
                ) = (
                    self.ack_socket.recv_multipart()
                )  # wait for consumer acknowledgement

                batch_max = int(batch_max)
                consumer_index = str(consumer_index)

                logger.info(
                    f"Consumer: {consumer_index}, "
                    f"batch count: {batch_max}, "
                    f"total batches: {self.data_loader_len}"
                )

                if consumer_index in expected:
                    expected.remove(consumer_index)

                if accepted == b"1":
                    self.consumers[consumer_index].add_batch(batch_max, payload)

            else:
                logger.info(
                    "Timeout on Ack, assuming consumer is dead",
                    len(self.hb.consumers),
                    expected,
                )
                expected = {}

    def __len__(self) -> int:
        """Get the length of the data loader.

        Returns:
            The number of batches in the data loader.
        """
        return self.data_loader_len

    def _send_consumer_len(self) -> None:
        """Send dataset metadata to newly connected consumers.

        Transmits:
        - Total number of batches in dataset
        - Maximum allowed buffer size
        """
        logger.info("Sending data loader length")
        self.socket.send_pyobj(
            {
                "data_loader_len": self.__len__(),
                "max_buffer_size": self.consumer_max_buffer_size,
                "loader_batch_size": self.loader_batch_size,
            }
        )
