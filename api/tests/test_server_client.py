import multiprocessing
import os
import sys
import time

import requests
import uvicorn

from tests import requester

# , Remove reload arg due to conflict between reload and uvicorn.

def start_uvicorn():
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", 8000)
    # , Remove reload arg due to conflict between reload and uvicorn.
    uvicorn.run("apis.main_api_db:app", host=host, port=int(port))


def wait_for_server(host, port, timeout):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


def run_server_client(timeout):
    # Start Uvicorn in a separate thread
    uvicorn_process = multiprocessing.Process(target=start_uvicorn)
    uvicorn_process.start()

    # Make a request.
    # Wait for the server to start up
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    if wait_for_server(host, port, timeout):
        try:
            # Run the test requests
            requester.main()
        finally:
            # Terminate the server process
            uvicorn_process.terminate()
            uvicorn_process.join()
    else:
        uvicorn_process.terminate()
        uvicorn_process.join()
        raise RuntimeError("Server failed to start within the timeout period")


def test_server_client(args=None):
    run_server_client(timeout=20)


if __name__ == "__main__":
    test_server_client(sys.argv[1:])
