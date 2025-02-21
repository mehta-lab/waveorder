import os, json
from pathlib import Path
import socket
import submitit
import threading, time

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PATH = os.path.join(DIR_PATH, "main.py")

SERVER_PORT = 8089 # Choose an available port
JOBS_TIMEOUT = 5 # 5 mins
SERVER_uIDsjobIDs = {} # uIDsjobIDs[uid][jid] = job

class JobsManagement():
    
    def __init__(self, *args, **kwargs):
        self.clientsocket = None
        self.uIDsjobIDs = {} # uIDsjobIDs[uid][jid] = job        
        self.DATA_QUEUE = []
        
    def check_for_jobID_File(self, jobID, logs_path, extension="out"):

        if Path(logs_path).exists():
            files = os.listdir(logs_path)
            try:
                for file in files:
                    if file.endswith(extension):
                        if jobID in file:
                            file_path = os.path.join(logs_path, file)
                            f = open(file_path, "r")
                            txt = f.read()                    
                            f.close()
                            return txt
            except Exception as exc:
                print(exc.args)
        return ""
    
    def set_shorter_timeout(self):
        self.clientsocket.settimeout(30)
    
    def start_client(self):
        try:
            self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.clientsocket.settimeout(300)
            self.clientsocket.connect(('localhost', SERVER_PORT))
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
                            if len(data)>0:
                                decoded_string = data.decode()
                                json_str = str(decoded_string)
                                json_obj = json.loads(json_str)
                                u_idx = json_obj["uID"]
                                job_idx = str(json_obj["jID"])
                                cmd = json_obj["command"]
                                if cmd == "clientRelease":
                                    if self.has_submitted_job(u_idx, job_idx):
                                        self.clientsocket.close()
                                        break
                                if cmd == "cancel":
                                    if self.has_submitted_job(u_idx, job_idx):
                                        try:
                                            job = self.uIDsjobIDs[u_idx][job_idx]
                                            job.cancel()
                                        except Exception as exc:
                                            pass # possibility of throwing an exception based on diff. OS
                forDeletions = []
                for uID in self.uIDsjobIDs.keys():
                    for jID in self.uIDsjobIDs[uID].keys():
                        job = self.uIDsjobIDs[uID][jID]
                        if job.done():
                            forDeletions.append((uID, jID))
                for idx in range(len(forDeletions)):
                    del self.uIDsjobIDs[forDeletions[idx][0]][forDeletions[idx][1]]
                forDeletions = []
                for uID in self.uIDsjobIDs.keys():
                    if len(self.uIDsjobIDs[uID].keys()) == 0:                        
                        forDeletions.append(uID)
                for idx in range(len(forDeletions)):
                    del self.uIDsjobIDs[forDeletions[idx]]
                if len(self.uIDsjobIDs.keys()) == 0:
                    self.clientsocket.close()
                    break
        except Exception as exc:
            self.clientsocket.close()
            print(exc.args)

    def check_all_ExpJobs_completion(self, uID):
        if uID in SERVER_uIDsjobIDs.keys():
            for jobEntry in SERVER_uIDsjobIDs[uID].keys():
                job:submitit.Job = SERVER_uIDsjobIDs[uID][jobEntry]["job"]
                jobBool = SERVER_uIDsjobIDs[uID][jobEntry]["bool"]
                if job is not None and job.done() == False:
                    return False
                if jobBool == False:
                    return False
        return True

    def put_Job_completion_in_list(self, job_bool, uID: str, jID: str, mode="client"):
        if uID in SERVER_uIDsjobIDs.keys():
            if jID in SERVER_uIDsjobIDs[uID].keys():
                SERVER_uIDsjobIDs[uID][jID]["bool"] = job_bool
    
    def add_data(self, data):
        self.DATA_QUEUE.append(data)
    
    def send_data_thread(self):
        thread = threading.Thread(target=self.send_data)
        thread.start()

    def send_data(self):
        data = "".join(self.DATA_QUEUE)
        self.clientsocket.send(data.encode())
        self.DATA_QUEUE = []

    def put_Job_in_list(self, job, uID: str, jID: str, well:str, log_folder_path:str="", mode="client"):
        try:
            well = str(well)
            jID = str(jID)
            if ".zarr" in well:
                wells = well.split(".zarr")
                well = wells[1].replace("\\","-").replace("/","-")[1:]
            if mode == "client":
                if uID not in self.uIDsjobIDs.keys():
                    self.uIDsjobIDs[uID] = {}
                    self.uIDsjobIDs[uID][jID] = job
                else:
                    if jID not in self.uIDsjobIDs[uID].keys():
                        self.uIDsjobIDs[uID][jID] = job
                json_obj = {uID:{"jID": str(jID), "pos": well, "log": log_folder_path}}
                json_str = json.dumps(json_obj)+"\n"                
                self.add_data(json_str)
            else:
                # from server side jobs object entry is a None object
                # this will be later checked as completion boolean for a ExpID which might
                # have several Jobs associated with it
                if uID not in SERVER_uIDsjobIDs.keys():
                    SERVER_uIDsjobIDs[uID] = {}
                    SERVER_uIDsjobIDs[uID][jID] = {}
                    SERVER_uIDsjobIDs[uID][jID]["job"] = job
                    SERVER_uIDsjobIDs[uID][jID]["bool"] = False
                else:                    
                    SERVER_uIDsjobIDs[uID][jID] = {}
                    SERVER_uIDsjobIDs[uID][jID]["job"] = job
                    SERVER_uIDsjobIDs[uID][jID]["bool"] = False
        except Exception as exc:
            print(exc.args)
    
    def has_submitted_job(self, uID: str, jID: str, mode="client")->bool:
        jID = str(jID)
        if mode == "client":
            if uID in self.uIDsjobIDs.keys():
                if jID in self.uIDsjobIDs[uID].keys():
                    return True
            return False
        else:
            if uID in SERVER_uIDsjobIDs.keys():
                if jID in SERVER_uIDsjobIDs[uID].keys():
                    return True
            return False
