import subprocess
import psutil
import threading, sys

class Print:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(text):
        print(f"{Print.OKBLUE}{text}{Print.ENDC}")

    @staticmethod
    def info2(text):
        print(f"{Print.OKCYAN}{text}{Print.ENDC}")

    @staticmethod
    def success(text):
        print(f"{Print.OKGREEN}{text}{Print.ENDC}")

    @staticmethod
    def warning(text):
        print(f"{Print.WARNING}{text}{Print.ENDC}")

    @staticmethod
    def fail(text):
        print(f"{Print.FAIL}{text}{Print.ENDC}")


def kill_process_if_running(PORT):
    for conn in psutil.net_connections():
        if conn.laddr.port == PORT:
            process = psutil.Process(conn.pid)
            if "python3.11" in process.name():
                process.kill()
            else:
                print("The specified port is in use. Please resolve this conflict or use another port for service.")
                return False
    
    return True


def _redir_server_output(proc, wait_for_text, child_finished):
    for line in iter(proc.stdout.readline, b''):
        sys.stderr.write(line)
        sys.stderr.flush()
        if wait_for_text in line and not child_finished.is_set():
            Print.success("Server Started! Continuing with tests...")
            child_finished.set()


def run_server(wait_for_text="Starting production HTTP BentoServer", pipe_output=True):

    # Run 'bentoml serve .' command and pipe output to stdout
    server_process = subprocess.Popen(['bentoml', 'serve', 'test:ConvoService'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Start a thread to read and print the output of the subprocess
    if pipe_output:
        child_finished = threading.Event()
        output_thread = threading.Thread(target=_redir_server_output, args=(server_process, wait_for_text, child_finished))
        output_thread.start()
        child_finished.wait()
    else:
        while True:
            output = server_process.stdout.readline().strip()
            if output == '' and server_process.poll() is not None:
                break
            if output:
                print(output)
                if wait_for_text in output:
                    Print.success("Server Started! Continuing with tests...")
                    break

    return server_process