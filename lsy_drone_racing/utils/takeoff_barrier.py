"""Synchronized takeoff barrier for multi-drone coordination.

This module provides a TakeOffBarrier class that coordinates the start of multiple drones
across processes or machines. All drones wait on the barrier during environment reset,
ensuring synchronized takeoff.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import socket
import tempfile
import time
from multiprocessing.managers import SyncManager
from pathlib import Path
from threading import BrokenBarrierError

logger = logging.getLogger(__name__)


class TakeOffBarrier:
    """Manages a shared multiprocessing barrier for synchronized drone takeoff.

    This class handles barrier creation, discovery, and coordination across processes.
    It supports both local and cross-machine setups through environment variables and
    shared metadata files.
    """

    def __init__(
        self,
        authkey: bytes,
        timeout_s: float,
        filename: str,
        port: int,
        bind_host: str = "0.0.0.0",
        public_host: str | None = None,
    ):
        """Initialize the TakeOffBarrier.

        Args:
            authkey: Authentication key for the barrier manager.
            timeout_s: Timeout in seconds for barrier wait.
            filename: Filename for metadata storage.
            port: Port for the barrier manager.
            bind_host: Interface to bind to (default: 0.0.0.0 for all interfaces).
            public_host: Public IP address for cross-machine setup. If None, auto-detect.
        """
        self.authkey = authkey
        self.timeout_s = timeout_s
        self.filename = filename
        self.port = port
        self.bind_host = bind_host
        self.public_host = public_host or self._get_public_address()

        self._barrier_manager: SyncManager | None = None
        self._owns_barrier_manager = False

    def _barrier_meta_path(self) -> Path:
        """Get the path to the barrier metadata file."""
        cache_dir = Path(tempfile.gettempdir()) / "lsy_drone_racing"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / self.filename

    def _read_barrier_meta(self) -> dict | None:
        """Read barrier metadata from file."""
        path = self._barrier_meta_path()
        try:
            with path.open("r", encoding="ascii") as file:
                return json.load(file)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            logger.warning("Barrier metadata file is invalid; ignoring it and creating a new one.")
            return None

    def _write_barrier_meta(self, meta: dict):
        """Write barrier metadata to file."""
        path = self._barrier_meta_path()
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="ascii") as file:
            json.dump(meta, file)
        tmp_path.replace(path)

    @staticmethod
    def _get_public_address() -> str:
        """Get the public/local IP address that would be used to reach external networks."""
        try:
            # Connect a UDP socket to a public IP without sending data to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            s.close()
            return addr
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"

    def _resolve_barrier_hosts(self) -> tuple[str, str, int]:
        """Return bind host, public host, and port for the barrier manager."""
        return self.bind_host, self.public_host, self.port

    def _create_barrier_manager(self, parties: int) -> mp.managers.ProxyBase:
        """Create a new barrier manager and register metadata."""
        class _BarrierManager(SyncManager):
            pass

        barrier_cache: dict[str, mp.Barrier] = {}

        def _get_barrier():
            if "barrier" not in barrier_cache:
                barrier_cache["barrier"] = mp.Barrier(parties)
            return barrier_cache["barrier"]

        _BarrierManager.register("get_barrier", callable=_get_barrier, exposed=("wait",))
        bind_host, public_host, port = self._resolve_barrier_hosts()
        manager = _BarrierManager(address=(bind_host, port), authkey=self.authkey)
        manager.start()
        meta = {
            "host": public_host,
            "port": port,
            "authkey": self.authkey.hex(),
            "parties": parties,
        }
        self._write_barrier_meta(meta)
        self._barrier_manager = manager
        self._owns_barrier_manager = True
        logger.info(
            "Created start barrier manager at %s:%d (bind: %s)",
            public_host,
            port,
            bind_host,
        )
        return manager.get_barrier()

    def _connect_to_barrier(self, meta: dict) -> mp.managers.ProxyBase:
        """Connect to an existing barrier manager."""
        class _BarrierManager(SyncManager):
            pass

        _BarrierManager.register("get_barrier", exposed=("wait",))
        authkey = bytes.fromhex(meta["authkey"])
        manager = _BarrierManager(address=(meta["host"], meta["port"]), authkey=authkey)
        manager.connect()
        self._barrier_manager = manager
        self._owns_barrier_manager = False
        return manager.get_barrier()

    def _get_or_create_barrier(self, parties: int) -> mp.managers.ProxyBase:
        """Get or create a shared barrier, handling discovery and creation logic."""
        barrier = None
        meta = self._read_barrier_meta()

        # Try metadata (works if shared over network storage)
        if barrier is None and meta is not None:
            try:
                barrier = self._connect_to_barrier(meta)
                logger.info(
                    "Connected to existing start barrier manager at %s:%d",
                    meta.get("host"),
                    meta.get("port"),
                )
                if meta.get("parties") and meta["parties"] != parties:
                    logger.warning(
                        "Barrier parties mismatch (meta=%s, current=%s); continuing anyway",
                        meta["parties"],
                        parties,
                    )
            except OSError as exc:
                logger.warning("Could not connect to barrier manager (%s); starting a new one.", exc)
                try:
                    self._barrier_meta_path().unlink()
                except FileNotFoundError:
                    pass

        if barrier is None:
            barrier = self._create_barrier_manager(parties)
            meta = self._read_barrier_meta()
            logger.info(
                "Started new start barrier manager at %s:%s",
                meta.get("host") if meta else "unknown",
                meta.get("port") if meta else "unknown",
            )

        return barrier

    def wait(self, rank: int, parties: int, participate: bool = True):
        """Wait on the barrier until all parties have arrived.

        Args:
            rank: Rank of the calling process.
            parties: Total number of parties expected at the barrier.
            participate: If True, wait on the barrier. If False, only create/connect to
                the barrier manager without participating (useful for manager-only hosts).

        Raises:
            RuntimeError: If the barrier times out or is broken.
        """
        barrier = self._get_or_create_barrier(parties)
        
        if not participate:
            logger.info("Barrier manager created for %d parties (not participating)", parties)
            return
        
        logger.info("Rank %d waiting on start barrier (%d parties)", rank, parties)
        t0 = time.perf_counter()
        try:
            barrier.wait(timeout=self.timeout_s)
        except BrokenBarrierError as exc:
            raise RuntimeError("Start barrier broken or timed out") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Rank %d passed start barrier in %.1f ms", rank, latency_ms)

    def close(self):
        """Shutdown the barrier manager if owned by this process."""
        if self._barrier_manager is not None and self._owns_barrier_manager:
            try:
                self._barrier_manager.shutdown()
                logger.info("Barrier manager shutdown complete")
            except Exception as exc:
                logger.warning("Error shutting down barrier manager: %s", exc)
