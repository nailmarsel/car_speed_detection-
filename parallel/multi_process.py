import os
import random
from multiprocessing import Process
from shutil import copy


def cmd(command: str):
    print('cmd command', command)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    os.system(command)



def func(counter, vid):
    py_file = f'python-files/rtdetr_{counter}.py'
    path_to_video = str(vid)
    path_to_json = str(json_paths) + '/' + str(vid).replace('.mp4', '.json').split('/')[-1]
    if counter % 8 == 1:
        counter += 1
    cmd_text = "python " + "'" + py_file + "'" + " '" + path_to_video + "'" + " '" + path_to_json + "'" + " '" + str(
        counter % 7) + "'"
    cmd(cmd_text)
    counter += 1


if __name__ == '__main__':
    path_to_py = '/home/ips/hackathon-2/rtdetr_test_snail.py'
    video_paths = '/home/ips/hackathon'
    json_paths = '/home/ips/hackathon'
    src_path = path_to_py
    for i in range(10):
        destination_path = f'python-files/rtdetr_{i}.py'
        copy(src_path, destination_path)

    vid_files = []

    for filename in os.listdir(video_paths):
        f = os.path.join(video_paths, filename)
        if os.path.isfile(f) and filename.endswith('.mp4'):
            vid_files.append(f)
        if len(vid_files) == 30:
            break
    counter = 1

    processes = []

    for vid in vid_files:
        if counter==7:
            counter=1
        elif counter==6:
            counter=7
        elif counter==5:
            counter=6
        elif counter==4:
            counter=5
        elif counter==3:
            counter=4
        elif counter==2:
            counter=3
        elif counter==1:
            counter=2
        if counter==1:
            cou=0
        else:
            cou=counter
        p = Process(target=func, args=(cou, vid))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
