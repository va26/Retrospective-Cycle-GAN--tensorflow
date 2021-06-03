import os
import subprocess
import glob

my_path = r".\\UCF-101\\vid_data"
files = glob.glob(my_path + r'\\*\\*.avi', recursive=True)


for vidItem in files:
    path = os.path.splitext(vidItem)[0]
    store_pth = path.replace("vid_data", "img_data")
    if not os.path.exists(store_pth):
        os.makedirs(store_pth)
    query = "ffmpeg -i " + vidItem + " " + store_pth + "\\pic%04d.jpg -hide_banner"
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    s = str(response).encode('utf-8')

