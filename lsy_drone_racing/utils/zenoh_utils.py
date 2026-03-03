"""Utilities for Zenoh-based host-client communication in multi-drone racing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import zenoh

logger = logging.getLogger(__name__)


@dataclass
class HostReadyMessage:
    """Message sent by host to indicate it's ready.
    
    Attributes:
        elapsed_time: Elapsed time since the race started (in IDLE/INITIALIZED, this is 0).
        timestamp: Timestamp when the message was sent (for latency measurement).
    """
    elapsed_time: float
    timestamp: float


@dataclass
class ClientReadyMessage:
    """Message sent by client in response to host ready message.
    
    Attributes:
        drone_rank: Rank of the drone.
        ready: Whether the client is ready.
        timestamp: Timestamp when the message was sent (for latency measurement).
    """
    drone_rank: int
    ready: bool
    timestamp: float


@dataclass
class RaceStartMessage:
    """Message sent by host to start the race.
    
    Attributes:
        elapsed_time: Elapsed time (initially 0, but sent periodically during operation).
        timestamp: Timestamp when the message was sent.
        finished: Whether the host has finished the race.
    """
    elapsed_time: float
    timestamp: float
    finished: bool = False


@dataclass
class ClientStateMessage:
    """Message sent by client during operation.
    
    Attributes:
        drone_rank: Rank of the drone.
        action: Control action (array or list).
        elapsed_time: Elapsed time (should match host's for latency measurement).
        timestamp: Timestamp when the message was sent.
        stopped: Whether the client has stopped (finished or error).
        next_gate_idx: Next gate index to visit (-1 if finished).
    """
    drone_rank: int
    action: list
    elapsed_time: float
    timestamp: float
    stopped: bool = False
    next_gate_idx: int = 0


@dataclass
class HostPingMessage:
    """Ping message from host to client for latency calibration.
    
    Attributes:
        drone_rank: Rank of the drone.
        host_timestamp: Timestamp when host sent this ping.
    """
    drone_rank: int
    host_timestamp: float


@dataclass
class ClientPongMessage:
    """Pong message from client back to host for latency calibration.
    
    Attributes:
        drone_rank: Rank of the drone.
        host_timestamp: Echo of the host's timestamp from the ping.
        client_timestamp: Timestamp when client sent this pong.
    """
    drone_rank: int
    host_timestamp: float
    client_timestamp: float


def serialize_message(message: Any) -> str:
    """Serialize a message dataclass to JSON.
    
    Args:
        message: Dataclass instance to serialize.
        
    Returns:
        JSON string representation of the message.
    """
    msg_dict = asdict(message)
    return json.dumps(msg_dict, default=str)


def deserialize_message(json_str: str, message_class: type) -> Any:
    """Deserialize a JSON string to a message dataclass.
    
    Args:
        json_str: JSON string representation.
        message_class: Target dataclass type.
        
    Returns:
        Deserialized message instance.
    """
    msg_dict = json.loads(json_str)
    # Convert action back to list if needed
    if "action" in msg_dict and isinstance(msg_dict["action"], (list, dict)):
        msg_dict["action"] = list(msg_dict["action"]) if isinstance(msg_dict["action"], (list, tuple)) else msg_dict["action"]
    return message_class(**msg_dict)


def compute_latency_ms(timestamp: float, clock_offset: float = 0.0) -> float:
    """Compute latency in milliseconds from a given timestamp.
    
    Args:
        timestamp: Original timestamp when message was sent.
        clock_offset: Optional clock offset between machines (in seconds) for correction.
                     Set this to the calibrated offset from ping-pong to correct for
                     clock skew between host and client machines.
        
    Returns:
        Latency in milliseconds.
    """
    return (time.perf_counter() - timestamp - clock_offset) * 1000


class ZenohPublisher:
    """Wrapper around Zenoh publisher for convenient publishing."""
    
    def __init__(self, session: zenoh.Session, key: str):
        """Initialize the publisher.
        
        Args:
            session: Zenoh session.
            key: Key expression to publish on.
        """
        self.session = session
        self.key = key
        self.publisher = session.declare_publisher(key)
        logger.info(f"Declared publisher on '{key}'")
    
    def publish(self, message: Any):
        """Publish a message.
        
        Args:
            message: Message dataclass to publish.
        """
        payload = serialize_message(message)
        self.publisher.put(payload)
    
    def close(self):
        """Close the publisher."""
        self.publisher.undeclare()


class ZenohSubscriber:
    """Wrapper around Zenoh subscriber for convenient subscribing."""
    
    def __init__(
        self,
        session: zenoh.Session,
        key: str,
        callback: Callable[[str], None],
    ):
        """Initialize the subscriber.
        
        Args:
            session: Zenoh session.
            key: Key expression to subscribe to.
            callback: Callback function to be called with payload when message received.
        """
        self.session = session
        self.key = key
        self.callback = callback
        
        def listener(sample: zenoh.Sample):
            try:
                payload = sample.payload.to_string()
                self.callback(payload)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")
        
        self.subscriber = session.declare_subscriber(key, listener)
        logger.info(f"Declared subscriber on '{key}'")
    
    def close(self):
        """Close the subscriber."""
        self.subscriber.undeclare()


def create_zenoh_session(conf: zenoh.Config | None = None) -> zenoh.Session:
    """Create and open a Zenoh session.
    
    Args:
        conf: Optional Zenoh configuration. If None, uses default config.
        
    Returns:
        Opened Zenoh session.
    """
    zenoh.init_log_from_env_or("error")
    if conf is None:
        conf = zenoh.Config()
    session = zenoh.open(conf)
    logger.debug("Zenoh session opened")
    return session
