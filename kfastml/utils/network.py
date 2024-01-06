import asyncio
import socket
import traceback
from asyncio import Future
from typing import Optional

import requests
import zmq

from kfastml import log


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


def download_from_uri(uri: str) -> Future[bytes | None]:
    # noinspection PyBroadException
    def _sync_bytesio_request(_uri: str) -> [bytes | None]:
        try:
            response = requests.get(uri)
            if response.status_code == 200:
                return response.content
        except:
            log.error(traceback.format_exc())
        return None

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _sync_bytesio_request, uri)
    return future
