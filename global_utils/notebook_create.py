import socket, os
import subprocess
from notebook.auth import passwd

def socket_port(ip, port):
    '''
    detect a port is used or not
    '''
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        if result == 0:
            return False
    except:
        print('port scan exception')
    return True

def find_container(container_name):
    """find_container find docker container exist or not

    Using docker inspect to find container exist or not. if find a container, docker inspect would return a long string, at least 100, otherwise return a short string.

    Args:
        container_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    p = subprocess.Popen(f"docker inspect {container_name}", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out = p.stdout.read()
    if len(str(out)) < 100:
        return False
    else:
        return True

if __name__ == '__main__':
    
    # assign port 
    # port = input('Input the port you want to assign: ')
    port = input('输入你想分配的notebook端口号: ')
    port = int(port)
    
    if not socket_port("127.0.0.1", port):
        print("端口已被占用，自动查找可用端口号...")
        temp = 1
        
        while not socket_port("127.0.0.1", port + temp):
            if temp == 100: break
            temp += 1
        
        if temp == 100:
            print("没有找到可用端口，请重新指定端口号")
            import sys
            sys.exit(0)
        else:
            print("查找到可用端口号：", port + temp)
            port = port + temp
            
    ssh_port = input('输入你想分配的ssh端口号: ')
    if not socket_port("127.0.0.1", port):
        print("端口已被占用，自动查找可用端口号...")
        temp = 1
        
        while not socket_port("127.0.0.1", ssh_port + temp):
            if temp == 100: break
            temp += 1
        
        if temp == 100:
            print("没有找到可用端口，请重新指定端口号")
            import sys
            sys.exit(0)
        else:
            print("查找到可用端口号：", ssh_port + temp)
            ssh_port = ssh_port + temp      
    
    
    # setting mount path
    group = input("输入你的组别(综合组：ecg，病理组：path，生信组：biod)：") 
    group = group.strip().lower()
    if group not in ['ecg','path','biod']:
        group = input("重新输入你的组别(综合组：ecg，病理组：path，生信组：biod)：").strip().lower()
    if group not in ['ecg','path','biod']:
        print("不存在该组别，退出程序")
        sys.exit(0)
        
    # setting container name    
    container_name = input("输入你的容器名：")
    if find_container(container_name):
        container_name = input("容器名重复，重新输入你的容器名：")
    if find_container(container_name):
        print("容器名重复，退出程序")
        sys.exit(0)
    
    # setting folder name 
    folder = input("输入你的文件夹名：")
    if not os.path.exists(f"/home/nas2/{group}/{folder}"):
        os.mkdir(f"/home/nas2/{group}/{folder}")
    
    
    # setting notebook password    
    if os.path.exists(f"/home/nas2/{group}/{folder}/startup.sh"):
        os.popen("echo '' > startup.sh")
    if os.path.exists(f"/home/nas2/{group}/{folder}/jupyter_server_config.json"):
        os.popen("echo '' > jupyter_server_config.json")
    password = input("输入你的notebook密码：")
    password = passwd(password, algorithm='sha1') 
    
    text = r'''#!/bin/bash
    pip install jupyterlab
    echo '' > /root/.jupyter/jupyter_lab_config.py
    rm /root/.jupyter/jupyter_lab_config.py
    jupyter lab --generate-config
    echo c.ServerApp.ip = \'*\' >> /root/.jupyter/jupyter_lab_config.py
    echo c.ServerApp.allow_remote_access = True >> /root/.jupyter/jupyter_lab_config.py
    echo c.ServerApp.root_dir = \'/\' >> /root/.jupyter/jupyter_lab_config.py
    '''
    
    config = r'''{
        "ServerApp": { ''' + "\n" + f'\"password\":\"{password}' + r'''"
            }
    }'''
    
    with open(f'/home/nas2/{group}/{folder}/jupyter_server_config.json', 'w') as f:
        for i in config.splitlines():
            f.write(i.lstrip() + "\n")
    
    with open(f"/home/nas2/{group}/{folder}/startup.sh", "w") as f:
        for i in text.splitlines():
            f.write(i.lstrip() + "\n")
        # f.write(r"c.ServerApp.token = 'ddf3b439292e0f0027e36fafa3f70df04bf79c98936cb7d4'" + "\n")
        f.write(f"cp /home/{folder}/jupyter_server_config.json /root/.jupyter/jupyter_server_config.json\n")
        f.write(r"jupyter lab --allow-root" + "\n")
    
    
    
    
    
    # print("你的token为：", "ddf3b439292e0f0027e36fafa3f70df04bf79c98936cb7d4")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    run_command = f"docker run -it -d --restart=always --runtime=nvidia -p {port}:8888 -p {ssh_port}:22 -v /home/nas2/{group}/datasets/:/data:ro -v /home/nas2/{group}/{folder}:/home/{folder}:rw --shm-size=4g --name {container_name} pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel sh /home/{folder}/startup.sh"   
         
    os.popen(run_command)
        