import pyttsx3
import threading
import queue
import time


class VoiceFeedback:

    def __init__(self, cooldown=2.0):

        self.cooldown = cooldown

        self.queue = queue.Queue()

        self.last_spoken = {}

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    # ========================================================
    # SPEAK
    # ========================================================

    def speak(self, message, key=None):

        if not message:
            return

        if key is None:
            key = message

        current_time = time.time()

        last_time = (
            self.last_spoken.get(key, 0)
        )

        # Prevent the same message from
        # being spoken repeatedly.
        if (
            current_time - last_time
            < self.cooldown
        ):
            return

        self.last_spoken[key] = (
            current_time
        )

        self.queue.put(message)

    # ========================================================
    # BACKGROUND WORKER
    # ========================================================

    def _worker(self):

        try:

            engine = pyttsx3.init()

            # Slightly slower than default
            # for clearer instructions.
            engine.setProperty(
                "rate",
                165
            )

            while self.running:

                try:

                    message = (
                        self.queue.get(
                            timeout=0.2
                        )
                    )

                except queue.Empty:

                    continue

                try:

                    engine.say(
                        message
                    )

                    engine.runAndWait()

                except Exception as e:

                    print(
                        "Voice error:",
                        e
                    )

                finally:

                    self.queue.task_done()

        except Exception as e:

            print(
                "Could not initialize "
                "voice engine:",
                e
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        try:

            self.queue.put_nowait("")

        except Exception:

            pass