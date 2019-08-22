import re
import os


def print_instructions():
    usr=os.environ['USER']
    with open('ip.txt') as f:
        myip = f.readline()

    myip=myip.rstrip()
    
    textfile = open('jupyter_logbook.txt', 'r')
    matches = []
    reg = re.compile('^\s*http://localhost:([0-9]*)')
    for line in textfile:
        match=reg.match(line)
        if match is not None:
            port=match.group(1)
            print("On your laptop open a new terminal and open an ssh tunnel by pasting this:")
            print('ssh -L {}:localhost:{} {}@{} -N -f'.format(port, port, usr, myip))
            print("NOTE: If you are using non-default keys you will need to point ssh client to your private key like this:")
            print('ssh -i <path_to/my_private_key> -L {}:localhost:{} {}@{} -N -f'.format(port, port, usr, myip))
            print("then in your browser address bar paste:")
            print(line.lstrip())
            
    textfile.close()




if __name__=='__main__':
    print_instructions()
