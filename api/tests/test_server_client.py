import os
import sys
import multiprocessing
import time

import uvicorn

from tests import requester


def start_uvicorn():
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", 8000)
    uvicorn.run("apis.main_api_db:app", host=host, port=int(port), reload=True)


def run_server_client():
    # Start Uvicorn in a separate thread
    uvicorn_process = multiprocessing.Process(target=start_uvicorn)
    uvicorn_process.start()

    # Make a request.
    time.sleep(10)
    requester.main()


def main(args=None):
    run_server_client()


if __name__ == "__main__":
    main(sys.argv[1:])
