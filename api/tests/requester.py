import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv
from path import Path


def run(fn_in, host="localhost", port=8000):
    load_dotenv()
    email = os.getenv('DEMO_USER_EMAIL')
    password = os.getenv('DEMO_USER_PASSWORD')

    n_rep = 1

    print("\nHealth Check")
    headers = {}
    try:
        response = requests.get(url=f"http://{host}:{port}/", headers=headers, )
    except Exception:
        raise ValueError("Not possible to connect")
    print(response)
    print(response.json())

    print("\nLogin for Access Token")
    response = requests.post(url=f'http://{host}:{port}/login_for_access_token/',
                             json={'email': email, 'password': password})
    print(response)
    print(response.json())
    token = response.json()['access_token']
    headers.update({'Authorization': f'Bearer {token}', })

    print("\nTest authorization")
    response = requests.get(url=f'http://{host}:{port}/protected_route/', headers=headers)
    print(response)
    print(response.json())

    print("\nis_voice_ai_gen: 300Kb audio file")
    start_time = time.perf_counter()
    for _ in range(n_rep):
        response = requests.post(url=f"http://{host}:{port}/is_voice_ai_gen/",
                                 files={"audio_file": open(fn_in, "rb")}, headers=headers, )
    time_elapse = 1000 * (time.perf_counter() - start_time) / n_rep
    print(f"Time_elapse={time_elapse}ms")
    print(response)
    print(response.json())


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fn_in', default="tests/tmp_data/2063112ff5b3ef954832985c2b289e729df04fdedd78da40963229bed71f83c9.mp3", type=Path,
                        help='Input file checkpoint')
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == '__main__':
    main(sys.argv[1:])
