import threading

from PySide6.QtCore import QObject, Signal

from .process import Cancellation, Cancelled


class Worker(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()
    cancelled = Signal()

    def __init__(self, function, token: Cancellation, parent=None):
        super().__init__(parent)
        self.function = function
        self.token = token
        self.thread = threading.Thread(target=self.run, name="ClipnadoWorker", daemon=False)

    def start(self):
        self.thread.start()

    def wait(self):
        self.thread.join()

    def run(self):
        try:
            value = self.function(self.token, self.progress.emit)
            self.token.check()
            self.result.emit(value)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()
