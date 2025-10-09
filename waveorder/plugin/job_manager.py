import subprocess


class JobManager:
    """This class manages the reconstruction jobs that are assigned via the GUI"""

    def __init__(self):
        self.jobs = {}  # uID -> Popen

    def run_job(self, uID, cmd, on_output=None, on_done=None):
        if uID in self.jobs:
            raise ValueError(f"Job {uID} already running.")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.jobs[uID] = proc

        def monitor():
            for line in proc.stdout:
                if on_output:
                    on_output(uID, line.strip())

            proc.wait()
            if on_done:
                on_done(uID, proc.returncode)
            del self.jobs[uID]

        monitor()
        # threading.Thread(target=monitor, daemon=True).start()

    def cancel_job(self, uID):
        proc: subprocess.Popen = self.jobs.get(uID)
        if proc and proc.poll() is None:
            proc.terminate()

    def is_running(self, uID):
        proc: subprocess.Popen = self.jobs.get(uID)
        return proc and proc.poll() is None
