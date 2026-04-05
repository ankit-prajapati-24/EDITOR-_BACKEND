from proglog import ProgressBarLogger
import asyncio

class SocketLogger(ProgressBarLogger):
    def __init__(self, sio, task_id):
        super().__init__()
        self.sio = sio
        self.task_id = task_id

    def callback(self, **changes):
        print(f"SocketLogger callback for task {self.task_id}: {changes}")
        if 'progress' in changes:
            percent = int(changes['progress'] * 100)

            # async emit
            asyncio.create_task(
                self.sio.emit('progress', {
                    'task_id': self.task_id,
                    'percent': percent
                })
            )