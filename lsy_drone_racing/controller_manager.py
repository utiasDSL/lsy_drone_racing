"""Asynchronous controller manager for multi-process control of multiple drones.

This module provides a controller manager that allows multiple controllers to run in separate
processes without blocking other controllers or the main process.
"""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

    from numpy.typing import NDArray


class ControllerManager:
    """Multi-process safe manager class for asynchronous/non-blocking controller execution.

    Note:
        The controller manager currently does not support step and episode callbacks.

    Todo:
        Implement an automated return mechanism for the controllers.
    """

    def __init__(self, controllers: list[Controller], default_action: NDArray):
        """Initialize the controller manager."""
        assert all(isinstance(c, Controller) for c in controllers), "Invalid controller type(s)!"
        self._controllers_cls = controllers
        self._obs_queues = [mp.Queue(1) for _ in controllers]
        self._action_queues = [mp.Queue(1) for _ in controllers]
        self._ready = [mp.Event() for _ in controllers]
        self._shutdown = [mp.Event() for _ in controllers]
        self._actions = np.tile(default_action, (len(controllers), 1))

    def start(self, init_args: tuple | None = None, init_kwargs: dict | None = None):
        """Start the controller manager."""
        for i, c in enumerate(self._controllers_cls):
            args = (
                c,
                tuple() if init_args is None else init_args,
                dict() if init_kwargs is None else init_kwargs,
                self._obs_queues[i],
                self._action_queues[i],
                self._ready[i],
                self._shutdown[i],
            )
            self._controller_procs.append(mp.Process(target=self._control_loop, args=args))
            self._controller_procs[-1].start()
        for ready in self._ready:  # Wait for all controllers to be ready
            ready.wait()

    def update_obs(self, obs: dict, info: dict):
        """Pass the observation and info updates to all controller processes.

        Args:
            obs: The observation dictionary.
            info: The info dictionary.
        """
        for obs_queue in self._obs_queues:
            _clear_producing_queue(obs_queue)
            obs_queue.put((obs, info))

    def latest_actions(self) -> NDArray:
        """Get the latest actions from all controllers."""
        for i, action_queue in enumerate(self._action_queues):
            if not action_queue.empty():  # Length of queue is 1 -> action is ready
                # The action queue could be cleared in between the check and the get() call. Since
                # the controller processes immediately put the next action into the queue, this
                # minimum block time is acceptable.
                self._actions[i] = action_queue.get()
        return np.array(self._actions)

    @staticmethod
    def _control_loop(
        cls: type[Controller],
        init_args: tuple,
        init_kwargs: dict,
        obs_queue: mp.Queue,
        action_queue: mp.Queue,
        ready: Event,
        shutdown: Event,
    ):
        controller = cls(*init_args, **init_kwargs)
        ready.set()
        while not shutdown.is_set():
            obs, info = obs_queue.get()  # Blocks until new observation is available
            action = controller.compute_control(obs, info)
            _clear_producing_queue(action_queue)
            action_queue.put_nowait(action)


def _clear_producing_queue(queue: mp.Queue):
    """Clear the queue if it is not empty and this process is the ONLY producer.

    Warning:
        Only works for queues with a length of 1.
    """
    if not queue.empty():  # There are remaining items in the queue
        try:
            queue.get_nowait()
        except Empty:  # Another process could have consumed the last item in between
            pass  # This is fine, the queue is empty
