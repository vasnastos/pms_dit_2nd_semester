from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import pickle


def pbe_password(pwd:str,filename=None):
    try:
        salt=b'\x00'
        # salt=os.urandom(16)
        kdf=PBKDF2HMAC(hashes.SHA1(),16,salt,1000,default_backend())
        key=kdf.derive(bytes(pwd,'UTF-8'))

        if filename:
            with open(filename,'wb') as WF:
                WF.write(key)

            with open(filename,'rb') as reader:
                print(reader.readlines())
        
            with open('key_save.pcl','wb') as writer:
                pickle.dump(key,writer)

            with open('key_save.pcl','rb') as reader:
                print(pickle.load(reader))

        else:
            print('Filename is setted to:None')
            print(f'Key:{key.hex()}')        

    except Exception as e:
        print(f'Invalid input value:{e}')


if __name__=='__main__':
    # pbe_password('HGF1234!@!!**!')
    pbe_password('@!keyTest1999','key_save')
