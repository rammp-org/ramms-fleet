"""Minimal client for URLab's RPC bridge (ZMQ REQ/REP, msgpack).

Covers what the fleet backend needs: the handshake, editor scene ops, Play In
Editor, and direct-mode stepping. The full protocol is in RAMMS at
Plugins/unreal-robotics-lab/docs/reference/protocol.md.
"""

from __future__ import annotations

import time
from typing import Any

import msgpack
import zmq


class BridgeError(RuntimeError):
    def __init__(self, op: str, reply: dict):
        self.code = reply.get("code", "")
        super().__init__(f"{op} failed: {self.code}: {reply.get('message', reply)}")


class URLabBridge:
    def __init__(self, address: str = "tcp://127.0.0.1:5559", timeout_s: float = 30.0):
        self.address = address
        self.timeout_ms = int(timeout_s * 1000)
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self.session_id: str | None = None
        self.handshake: dict = {}
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.address)

    def request(self, op: str, **fields: Any) -> dict:
        """Sends one op and returns its reply, following async editor jobs to completion."""
        message = {"op": op, **fields}
        if self.session_id and op != "hello":
            message["session_id"] = self.session_id
        try:
            self._socket.send(msgpack.packb(message, use_bin_type=True))
            reply = msgpack.unpackb(self._socket.recv(), raw=False)
        except zmq.Again as exc:
            # A REQ socket that missed a reply is stuck; start a fresh one.
            self._connect()
            raise TimeoutError(f"{op}: no reply from {self.address} within {self.timeout_ms} ms") from exc
        if reply.get("op") == "op_started":
            reply = self._wait_for_job(op, reply["job_id"])
        if reply.get("op") == "error":
            raise BridgeError(op, reply)
        return reply

    def _wait_for_job(self, op: str, job_id: str, poll_s: float = 0.25) -> dict:
        deadline = time.monotonic() + max(60.0, self.timeout_ms / 1000)
        while time.monotonic() < deadline:
            status = self.request("op_status", job_id=job_id)
            if status.get("state") in ("done", "failed"):
                return status.get("result", status)
            time.sleep(poll_s)
        raise TimeoutError(f"{op}: job {job_id} still running")

    def hello(self, observations: str = "standard") -> dict:
        self.session_id = None
        self.handshake = self.request("hello", observations=observations, include_assets=False)
        self.session_id = self.handshake["session_id"]
        return self.handshake

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
