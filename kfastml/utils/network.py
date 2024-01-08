import asyncio
import socket
import traceback
from asyncio import Future
from typing import Optional

import requests
import zmq

from kfastml import log


def get_unused_network_ports(count: int, start_range: int = 8000, exclude_map: dict = None) -> list[int]:
    assert count > 0
    unused_ports = []
    for port in range(start_range, 65535):
        if count <= 0:
            break
        if exclude_map and port in exclude_map:
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


def download_uris_async(uris: list[str]) -> list[Future[bytes | None]]:
    # noinspection PyBroadException
    def _sync_bytesio_request(_uri: str) -> [bytes | None]:
        try:
            response = requests.get(_uri)
            if response.status_code == 200:
                return response.content
        except:
            log.error(traceback.format_exc())
        return None

    def _download_uri(uri: str) -> Future[bytes | None]:
        loop = asyncio.get_event_loop()
        submit_future = loop.run_in_executor(None, _sync_bytesio_request, uri)
        return submit_future

    return [_download_uri(uri) for uri in uris]


def download_uris(uris: list[str]) -> list[bytes | None]:
    futures = download_uris_async(uris)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*futures))
    return list(results)
