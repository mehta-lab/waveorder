import json
import os
import socket
import threading
import time
from pathlib import Path

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PATH = os.path.join(DIR_PATH, "main.py")

SERVER_PORT = 65432  # Choose an available port
JOBS_TIMEOUT = 5  # 5 mins
SERVER_uIDs = {}  # uIDsjobIDs[uid] = bool


class JobsManagement:

    def __init__(self, *args, **kwargs):
        self.clientsocket = None
        self.uIDs = {}  # uIDsjobIDs[uid] = bool
        self.DATA_QUEUE = []
        self.do_print = True

    def set_shorter_timeout(self):
        self.clientsocket.settimeout(30)

    def start_client(self):
        try:
            self.clientsocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            self.clientsocket.settimeout(300)
            self.clientsocket.connect(("localhost", SERVER_PORT))
            self.clientsocket.settimeout(None)

            thread = threading.Thread(target=self.stop_client)
            thread.start()
        except Exception as exc:
            print(exc.args)

    # The stopClient() is called right with the startClient() but does not stop
    # and essentially is a wait thread listening and is triggered by either a
    # connection or timeout. Based on condition triggered by user, reconstruction
    # completion or errors the end goal is to close the socket connection which
    # would let the CLI exit. I could break it down to 2 parts but the idea was to
    # keep the clientsocket.close() call within one method to make it easier to follow.
    def stop_client(self):
        try:
            time.sleep(2)
            while True:
                time.sleep(1)
                buf = ""
                try:
                    buf = self.clientsocket.recv(1024)
                except:
                    pass
                if len(buf) > 0:
                    if b"\n" in buf:
                        dataList = buf.split(b"\n")
                        for data in dataList:
                            if len(data) > 0:
                                decoded_string = data.decode()
                                json_str = str(decoded_string)
                                json_obj = json.loads(json_str)
                                u_idx = json_obj["uID"]
                                cmd = json_obj["command"]
                                if cmd == "clientRelease":
                                    if self.has_submitted_job(u_idx):
                                        self.clientsocket.close()
                                        break
                                if cmd == "cancel":
                                    if self.has_submitted_job(u_idx):
                                        try:
                                            pass  # ToDo: Implement cancelling logic
                                        except Exception as exc:
                                            pass  # possibility of throwing an exception based on diff. OS
                forDeletions = []
                for uID in self.uIDs.keys():
                    jobBool = self.uIDs[uID]
                    if jobBool:
                        forDeletions.append(uID)
                for idx in range(len(forDeletions)):
                    del self.uIDs[forDeletions[idx]]
                if len(self.uIDs.keys()) == 0:
                    self.clientsocket.close()
                    break
        except Exception as exc:
            self.clientsocket.close()
            print(exc.args)

    def check_all_ExpJobs_completion(self, uID):
        if uID in SERVER_uIDs.keys():
            jobBool = SERVER_uIDs[uID]
            return jobBool
        return True

    def put_Job_completion_in_list(self, uID: str, finished, mode="client"):
        if uID in SERVER_uIDs.keys():
            SERVER_uIDs[uID] = finished
        if uID in self.uIDs.keys():
            self.uIDs[uID] = finished

    def isCompleted(self, uID: str):
        if uID in SERVER_uIDs.keys():
            return SERVER_uIDs[uID]
        if uID in self.uIDs.keys():
            return self.uIDs[uID]
        return False

    def add_data(self, data):
        self.DATA_QUEUE.append(data)

    def send_data_thread(self, data):
        thread = threading.Thread(target=self.send_data, args=(data,))
        thread.start()

    def send_data(self, data):
        # print("Client:" + data)
        self.clientsocket.send(data.encode())

    def put_Job_in_list(
        self,
        uID: str,
        msg: str = "",
        mode="client",
    ):
        try:
            if mode == "client":
                if uID not in self.uIDs.keys():
                    self.uIDs[uID] = False
                json_obj = {uID: {"msg": msg}}
                json_str = json.dumps(json_obj) + "\n"
                self.send_data_thread(json_str)
            else:
                # from server side jobs object entry is a None object
                # this will be later checked as completion boolean for a ExpID which might
                # have several Jobs associated with it
                if uID not in SERVER_uIDs.keys():
                    SERVER_uIDs[uID] = False
        except Exception as exc:
            print(exc.args)

    def has_submitted_job(self, uID: str, mode="client") -> bool:
        if mode == "client":
            if uID in self.uIDs.keys():
                return True
            return False
        else:
            if uID in SERVER_uIDs.keys():
                return True
            return False
