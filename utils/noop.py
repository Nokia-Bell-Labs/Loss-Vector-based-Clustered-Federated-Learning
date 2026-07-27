# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from utils.logging_utils import logging


class NoOpClass:
    """
    A class that performs no operations (no-op) for any method or attribute access.

    This class is designed to handle any method call or nested attribute access by
    logging a message indicating that a no-op was applied for the called method.

    Attributes:
        _name (str): The full name of the method being accessed.
        _logger (logging.Logger): The logger instance used for logging messages.

    Methods:
        __getattr__(name):
            Handles attribute access and returns a new instance of NoOpClass with the updated method name.

        __call__(*args, **kwargs):
            Handles method calls and logs a no-op message with the full method name.
    """

    def __init__(self, name='', logger=None):
        """
        Initializes a new instance of NoOpClass.

        Args:
            name (str): The full name of the method being accessed. Defaults to an empty string.
            logger (logging.Logger): The logger instance used for logging messages. Defaults to None.
        """
        self._name = name
        if logger is None:
            self._logger = logging.getLogger(__name__)  # Global logger, external to class
        else:
            self._logger = logger

    def __getattr__(self, name):
        """
        Handles attribute access and returns a new instance of NoOpClass with the updated method name.

        Args:
            name (str): The name of the accessed attribute.

        Returns:
            NoOpClass: A new instance of NoOpClass with the updated method name.
        """
        full_name = f"{self._name}.{name}" if self._name else name
        return NoOpClass(full_name, self._logger)

    def __call__(self, *args, **kwargs):
        """
        Handles method calls and logs a no-op message with the full method name.

        Args:
            *args: Positional arguments passed to the method.
            **kwargs: Keyword arguments passed to the method.
        """
        self._logger.debug(f"No-op: {self._name} - No operation applied.")
        return NoOpContextManager(self._name, self._logger)


class NoOpContextManager:
    """
    A context manager class that logs no-op messages for entering and exiting contexts.

    Attributes:
        _name (str): The full name of the method being accessed.
        _logger (logging.Logger): The logger instance used for logging messages.

    Methods:
        __enter__():
            Handles entering the context and logs a no-op message.

        __exit__(exc_type, exc_value, traceback):
            Handles exiting the context and logs a no-op message.
    """

    def __init__(self, name, logger):
        """
        Initializes a new instance of NoOpContextManager.

        Args:
            name (str): The full name of the method being accessed.
            logger (logging.Logger): The logger instance used for logging messages.
        """
        self._name = name
        self._logger = logger

    def __enter__(self):
        """
        Handles entering the context and logs a no-op message.

        Returns:
            NoOpContextManager: The current instance of NoOpContextManager.
        """
        self._logger.debug(f"No-op: {self._name} (entering context) - No operation applied.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Handles exiting the context and logs a no-op message.

        Args:
            exc_type: The exception type, if any.
            exc_value: The exception value, if any.
            traceback: The traceback, if any.
        """
        self._logger.debug(f"No-op: {self._name} (exiting context) - No operation applied.")


# Example usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger(__name__)
#
#     noop_instance = NoOpClass(logger=logger)
#
#     # Calling any method on the instance will log the no-op message with the method name
#     noop_instance.any_method()  # Logs: no op applied for any_method
#     noop_instance.any_module.any_function()  # Logs: no op applied for any_module.any_function
#     noop_instance.another_module.another_function('arg1', 'arg2')  # Logs: no op applied for another_module.another_function
#     noop_instance.yet_another_module.yet_another_function(key='value')  # Logs: no op applied for yet_another_module.yet_another_function
#
#     # Using the instance as a context manager
#     with noop_instance.any_method():
#         print("Inside the context")  # Logs: no op applied for any_method (entering context)
#         #       Inside the context
#         #       no op applied for any_method (exiting context)
