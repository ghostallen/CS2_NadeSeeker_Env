import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import deque
import threading
import time
import numpy as np
global_data = {"npos": []}

class GSIHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def do_POST(self):
        global stop_server
        if stop_server:  # discard requests immediately if stop_server is true
            return
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            gsi_payload = json.loads(post_data.decode('utf-8'))
            self._process_nade_infos(gsi_payload)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        return

    def _process_nade_infos(self, payload):
        global stop_server
        if 'grenades' in payload and not stop_server:  # ensure stop_server is false
            grenades_data = payload['grenades']
            if isinstance(grenades_data, dict):
                for gid, grenade_info in grenades_data.items():
                    position = grenade_info.get('position')
                    if position is not None:
                        logs.append(position)  # use logs defined at a higher scope
                        self.parse_log()
        return

    def parse_log(self):
        global stop_server, global_data
        if len(logs) >= 5 and logs.count(logs[0]) == len(logs):
            nade_log = logs[-1].split(', ')
            nade_log = [float(val) for val in nade_log]
            global_data["npos"] = nade_log  # update the global dictionary
            stop_server = True

def run_server_and_get_npos(host='127.0.0.1', port=23927, flt_compensate=0.2):
    start_time = time.time()
    global stop_server, logs, global_data
    stop_server = False
    logs = deque(maxlen=6)  # define logs at a higher scope
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, lambda *args, **kwargs: GSIHandler(*args, **kwargs))
    httpd.timeout = 0.1  # set timeout to 0.25 second
    
    start_time = time.time()
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()
    
    while server_thread.is_alive():
        if time.time() - start_time > 10:
            httpd._BaseServer__shutdown_request = True
            httpd.shutdown()
            httpd.server_close()
            return [np.nan, np.nan, np.nan, np.nan]

        if stop_server:  # check self.stop_server instead of httpd._BaseServer__shutdown_request
            httpd._BaseServer__shutdown_request = True
            httpd.shutdown()
            httpd.server_close()
    
    end_time = time.time()

    global_data["npos"].append(float(end_time - start_time + flt_compensate))

    return global_data["npos"]  # return the updated npos from the global dictionary


def get_npos(host='127.0.0.1', port=3000):
    return run_server_and_get_npos(host, port)