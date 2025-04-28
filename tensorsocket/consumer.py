import logging
import sys
import threading
import uuid
from queue import Queue
from typing import Tuple, Any, Iterator

import zmq

from .payload import TensorPayload
from .heartbeat import Heart

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    # level=logging.DEBUG,
    stream=sys.stdout,
)

logger = logging.getLogger("tensorsocket")
# logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://localhost"


def unpack(data: tuple) -> tuple:
    """Convert TensorPayload objects back to tensors.

    Args:
        data: Tuple containing possible TensorPayload objects

    Returns:
        Tuple with reconstructed tensors
    """
    if isinstance(data, dict):
        return {
            k: v.tensor if isinstance(v, TensorPayload) else v for k, v in data.items()
        }
    return tuple((t.tensor if isinstance(t, TensorPayload) else t for t in data))


class TensorConsumer:
    """Receives and processes tensor batches from remote producer.

    Handles:
    - Connection to producer
    - Batch receiving and unpacking
    - Progress tracking
    - Heartbeat monitoring
    """

    def __init__(
        self,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: tuple[int, int] = (4444, 4445),
        unpack_fn=unpack,
        batch_size: int = 64,
    ) -> None:
        """Initialize consumer connection.

        Args:
            port: Data reception port
            ack_port: Acknowledgement sending port
            heart_ports: (in, out) ports for heartbeat
            unpack_fn: Function to reconstruct tensors
        """
        self.unpack_fn = unpack_fn
        self.batch_size = batch_size

        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"{LOCALHOST}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.consumer_id = uuid.uuid4()

        # Ack
        self.ack_socket = self.context.socket(zmq.PUB)
        self.ack_socket.connect(f"{LOCALHOST}:{self.ack_port}")

        # Logic
        self.batch_count = 0
        self.batch_max = -1
        self.epoch = 0
        # self.receiving_epoch = 0

        # Heartbeat
        self.heart = Heart(
            self,
            self.consumer_id,
            self.batch_size,
            f"{LOCALHOST}:{self.heart_ports[0]}",
            f"{LOCALHOST}:{self.heart_ports[1]}",
        )
        self.heart.daemon = True
        self.heart.start()

        # On spawn, fetch payloads on socket until we get one with the data loader length
        while True:
            data = self.socket.recv_pyobj()
            if data.get("data_loader_len"):
                self.data_loader_len = data.get("data_loader_len")
                self.max_buffer_size = data.get("max_buffer_size")
                self.loader_batch_size = data.get("loader_batch_size")
                break

        # Buffer setup
        self.buffer = Queue(maxsize=self.max_buffer_size)
        self.fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.fetch_thread.start()

    def _fetch_loop(self) -> None:
        """Background thread for receiving batches.

        Continuously:
        1. Receives tensor data
        2. Handles special messages (length, stop)
        3. Processes regular batches
        4. Sends acknowledgements
        """
        while True:
            cuda_tensor_info = self.socket.recv_pyobj()

            if "data_loader_len" in cuda_tensor_info:
                continue

            if "stop_iteration" in cuda_tensor_info:
                self.buffer.put(cuda_tensor_info)
                continue

            if str(self.consumer_id) in cuda_tensor_info:  # Flexible
                messages = cuda_tensor_info[str(self.consumer_id)]
            elif "-1" in cuda_tensor_info and len(cuda_tensor_info) == 1:  # Static
                messages = cuda_tensor_info["-1"]
            else:  # Ignore
                messages = []

            received_new = False

            # print(messages)
            for message in messages:
                # print("MESSAGE", message.keys())
                if message["current_batch_index"] == self.batch_max + 1:
                    self.buffer.put(message)
                    self.batch_max = message["current_batch_index"]
                    received_new = True

            if received_new:
                self.ack_socket.send_multipart(
                    [
                        bytes(str(self.consumer_id).encode("utf-8")),
                        bytes(str(self.batch_max).encode("utf-8")),
                        b"1",
                    ]
                )
            else:
                self.ack_socket.send_multipart(
                    [
                        bytes(str(self.consumer_id).encode("utf-8")),
                        bytes(str(self.batch_max).encode("utf-8")),
                        b"0",
                    ]
                )

    def __iter__(self) -> Iterator:
        """Make consumer iterable for batch processing."""
        return self

    def __len__(self) -> int:
        """Get total number of batches in dataset."""
        return self.data_loader_len * self.batch_size // self.loader_batch_size

    def __next__(self) -> Tuple[int, Any]:
        """Get next batch from buffer.

        Returns:
            Tuple of (batch_index, tensor_data)

        Raises:
            StopIteration: At end of epoch
        """
        while True:
            payload = self.buffer.get()  # This will block if buffer is empty

            if "stop_iteration" in payload:
                self.batch_count = 0
                self.batch_max = -1
                if (
                    self.batch_count > 0
                ):  # Increase epoch if we have received at least one batch
                    self.epoch += 1
                    raise StopIteration
                else:
                    continue

            batch_idx = payload["current_batch_index"]

            batch = self.unpack_fn(payload["data"])

            # print("\n\n\n\n\n\n\n\nBATCH consumer!!!!!!!!!", batch_idx, batch)
            # exit()
            if batch_idx == self.batch_count:
                logger.info(
                    f"Epoch: {self.epoch}, batch_idx: {batch_idx}, batch count: {self.batch_count}"
                )
                self.batch_count += 1
                # return batch_idx, batch
                return batch
