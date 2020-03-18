"""Definitions of Qt signals for workers."""

from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    """
        Defines the signals available from a running worker thread.

        Supported signals are:

        finished
            `str` message to acknowledge the achievement

        progressed
            `int` percentage of actual progression

        finished
            `str` message to log

        error
            `str` error message to log

        result
            `object` data returned from processing, anything
    """

    finished = pyqtSignal(str)
    progressed = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    result = pyqtSignal(object)
