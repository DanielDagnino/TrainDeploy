import logging
import logging.config
import math
import multiprocessing
import os
import time

import requests
import uvicorn
from path import Path

from apis_tests import requester

# logging.basicConfig(level=logging.INFO)
logging.config.fileConfig('logging.conf')


def start_uvicorn():
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", 8000)
    logging.info(f"Starting server at {host}:{port}")

    uvicorn.run("apis.main_api_db:app",
                host=host, port=int(port),
                log_level="info", access_log=True)


def wait_for_server(host, port, timeout):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/")
            if response.status_code == 200:
                logging.info(f"Server started successfully at {host}:{port}")
                return True
        except requests.ConnectionError:
            time.sleep(1)
    logging.error(f"Server failed to start within {timeout} seconds")
    return False


def run_server_client(fns_lbls, timeout):
    # Test file exists.
    for fn, _ in fns_lbls:
        if not Path(fn).exists():
            raise FileExistsError(f'File {fn} does not exist')

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
            for fn, lbl in fns_lbls:
                score = requester.run(fn, host, port)
                assert math.fabs(score - lbl) < 0.1
        finally:
            # Terminate the server process
            uvicorn_process.terminate()
            uvicorn_process.join()
    else:
        uvicorn_process.terminate()
        uvicorn_process.join()
        raise RuntimeError(f'Server failed to start within the timeout period with "http://{host}:{port}/"')


def test_server_client():
    run_server_client(
        fns_lbls=[
            ["apis_tests/tmp_data/0f23e5875491a1d47e1afd6c0fd9eb9d78e899cc6c8cdd50b1ef0bf26b9664c1.mp3", 1],
            ["apis_tests/tmp_data/6918-61317-0016.flac", 0],
        ], timeout=60)


if __name__ == "__main__":
    test_server_client()
