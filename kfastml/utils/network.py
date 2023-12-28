import socket
from typing import Optional

import zmq


def get_unused_network_ports(count: int, start_range: int = 8000, exclude_map: dict = {}) -> list[int]:
    assert count > 0
    unused_ports = []
    for port in range(start_range, 65535):
        if count <= 0:
            break
        if port in exclude_map:
            continue
        with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s):
            s.settimeout(0)
            try:
                s.bind(("localhost", port))
                unused_ports.append(port)
                count -= 1
            except socket.error:
                pass
    return unused_ports


def recv_noblock_json(reply_socket: zmq.Socket) -> Optional[dict]:
    try:
        result = reply_socket.recv_json(flags=zmq.NOBLOCK)
        return result
    except zmq.ZMQError as err:
        if err.errno != zmq.EAGAIN:
            raise
    return None


def recv_noblock_pyobj(reply_socket: zmq.Socket) -> object:
    try:
        result = reply_socket.recv_pyobj(flags=zmq.NOBLOCK)
        return result
    except zmq.ZMQError as err:
        if err.errno != zmq.EAGAIN:
            raise
    return None
