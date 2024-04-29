import os
import sys
import threading
from time import time
import click

def execCmd(cmd):
    try:
        print("command %s start %s" % (cmd, time() ))
        os.system(cmd)
        print("command %s finish %s" % (cmd, time()))
    except Exception as e :     
       print('command %s\t failed \r\n%s' % (cmd,e))

@click.command()
@click.option('--input_dir', type=str, help='input bfm folder', default='')
@click.option('--save_dir', type=str, help='input bfm folder', default='')
@click.option('--n_thread', type=int, help='number of threads', default=20)
@click.option('--instance_per_thread', type=int, help='number of instances per thread', default=1)
def main(input_dir:str, save_dir:str, n_thread:int, instance_per_thread:int):
    
    root = input_dir
    threads = []

    print("start%s" % time())

    for g in range(n_thread):
        cmd = 'python mesh_convert_from_coeff.py --input_dir %s --save_dir %s --group %d --num %d'%(input_dir, save_dir, g, instance_per_thread)
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()          
        threads.append(th)

    for th in threads:
        th.join() 

    print("end%s" % time())

if __name__ == '__main__':
    main()

