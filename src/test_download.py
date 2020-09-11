import requests
import os

dir_path = "test_image/"


def download_image(url):
    file_name = url.split('/')[-1]+".jpg"
    out_path = os.path.join(dir_path, file_name)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    return out_path


download_image('https://www.wvxu.org/sites/wvxu/files/201904/raccoon-1905528_1920.jpg')
