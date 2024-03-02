import time

class PPosParser:

    def __init__(self, log_file, create_cfg=False, tp_cfg='None'):
        self.log_file = log_file
        self.create_cfg = create_cfg
        self.tp_cfg = tp_cfg

        return

    def is_float(self, n):
        negative = n.count('-') == 1 if n.startswith('-') else n.count('-') == 0
        dots = n.count('.') == 1 if '.' in n else True
        numeric = all([i.isdigit() for i in n.replace('.', '').replace('-', '')])
        return negative and dots and numeric

    def parse_log_line(self, log_line):
        log_line = log_line.replace(';setang', ' ;setang')

        # Remove timestamp; Minus the ppos_z by 64, which should be the exact ppos_z. 
        log_line_list = log_line.split(' ')[2:]  
        log_line_list[3] = str(float(log_line_list[3]) - 64)
        return log_line_list

    def floatlist(self, log_line_list):
        return [float(part) for part in log_line_list if self.is_float(part)]
    
    def write2cfg(self, list_str):
        with open(self.tp_cfg, 'w+', encoding='utf8') as f:
            f.write(' '.join(list_str) + '\n')  # Write log to file
        return

    def monitor_log_file(self):
        with open(self.log_file, 'r', encoding='utf-8') as file:
            # Go to the end of the file
            file.seek(0, 2)
            while True:
                line = file.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                else:
                    latest_log = line.strip()
                    if ';setang' in latest_log:
                        latest_log = self.parse_log_line(latest_log)

                        if self.create_cfg == True:
                            self.write2cfg(latest_log)

                        processed_log_line_list = self.floatlist(latest_log)

                        return processed_log_line_list
        

def get_ppos(log_file, create_cfg, tp_cfg):
    ppos_parser = PPosParser(log_file, create_cfg, tp_cfg)
    return ppos_parser.monitor_log_file()