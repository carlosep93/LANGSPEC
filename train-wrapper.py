from subprocess import Popen

p = Popen("./train-small.sh",cwd='/home/carlos/Documentos/interlingua-fairseq/', shell=True)
stdout, stderr = p.communicate()
