import time

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()

    n_rep = 1
    ip_address = "http://localhost:8000"
    headers = {}

    print("\nHealth Check")
    try:
        response = requests.get(url=f"{ip_address}/", headers=headers, )
    except Exception:
        raise ValueError("Not possible to connect")
    print(response)
    print(response.json())

    print("\nLogin for Access Token")
    email = 'johndoe@domain.ext'
    password = 'DEMO_dagnino_1234'
    response = requests.post(url=f'{ip_address}/login_for_access_token/', json={'email': email, 'password': password})
    print(response)
    print(response.json())
    token = response.json()['access_token']
    headers.update({'Authorization': f'Bearer {token}', })

    print("\nTest authorization")
    response = requests.get(url=f'{ip_address}/protected_route/', headers=headers)
    print(response)
    print(response.json())

    print("\nis_voice_ai_gen: 300Kb audio file")
    start_time = time.perf_counter()
    for _ in range(n_rep):
        response = requests.post(url=f"{ip_address}/is_voice_ai_gen/",
                                 files={"audio_file": open("tests/tmp_data/biden_yt_cut.mp3", "rb")}, headers=headers, )
    time_elapse = 1000 * (time.perf_counter() - start_time) / n_rep
    print(f"Time_elapse={time_elapse}ms")
    print(response)
    print(response.json())


if __name__ == "__main__":
    main()
