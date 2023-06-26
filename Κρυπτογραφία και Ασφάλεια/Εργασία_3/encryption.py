import os,argparse
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from datetime import datetime

class Encryptor:
    @staticmethod
    def read_bmp(bmp_file):
        image=None
        with open(bmp_file,'rb') as reader:
            image=reader.read()
        return image

    def __init__(self,parser):
        args=parser.parse_args()
        folder=args.folder
        self.password=os.path.join(folder,args.input) if folder else args.input
        self.key_file=os.path.join(folder,args.key) if folder else args.key
        self.output_file=os.path.join(folder,args.output) if folder else args.output
        self.config_file=os.path.join(folder,args.config) if folder else args.config
        self.feedback_mode=args.feedback
        self.algorithm=args.algorithm
        
        if args.bmp:
            self.password=Encryptor.read_bmp(self.password)
        
        self.iv_sizes={
            "AES":{
                "CBC":16,
                "OFB":16,
                "CFB": 16,
                "CTR": 16,
                "ECB": 0
            },
            "DES":{
                "CBC":8,
                "OFB":8,
                "CFB":8,
                "CTR":8,
                "ECB":0
            }
        }
        self.config=dict()

    def get_datetime(self):
        return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    def get_algorithm_mode_bytes(self):
        if self.algorithm in self.iv_sizes:
            if self.feedback_mode in self.iv_sizes[self.algorithm]:
                return self.iv_sizes[self.algorithm][self.feedback_mode]
        return 0

    def password_based_key_derivation(self,pwd:str):
        try:
            salt=os.urandom(16)
            kdf=PBKDF2HMAC(hashes.SHA1(),16,salt,1000,default_backend())
            key=kdf.derive(bytes(pwd,'UTF-8'))

            if self.key_file:
                with open(self.key_file,'wb') as WF:
                    WF.write(key)
            else:
                print('Filename is setted to:None')
                print(f'Key:{key.hex()}')        

        except Exception as e:
            print(f'Invalid input value:{e}')
        return key

    
    def generate_iv(self):
        iv_byte_length=self.iv_sizes.get(self.algorithm,None)
        if iv_byte_length:
            return os.urandom(iv_byte_length.get(self.feedback_mode,0))
        return None 
    
    def generate_config_file(self,initialization_vector):
        with open(self.config_file,'w') as writer:
            writer.write(f"Algorithm: {self.algorithm}\n")
            writer.write(f"Feedback Mode: {self.feedback_mode}\n")
            writer.write(f"IV: {initialization_vector.hex()}\n")


    def encrypt_password(self):
        password=None
        with open(self.password,'r') as reader:
            password=reader.readline().strip()
    
        print(f'[{self.get_datetime()}] Password inserted:{password}')
        key=self.password_based_key_derivation(password)
        print(f'[{self.get_datetime()}] Key derivation maded:{key.hex()}')
        print(f'[{self.get_datetime()}] Algorithm:{self.algorithm} Mode:{self.feedback_mode}')


        algorithm_object=None
        feedback_object=None
        iv=self.generate_iv()
        print(f'[{self.get_datetime()}] Initialization Vector:{iv}')
        self.generate_config_file(iv)

        if self.algorithm=="AES":
            algorithm_object=algorithms.AES(key)
        elif self.algorithm=="DES":
            algorithm_object=algorithms.TripleDES(key)
        
        if self.feedback_mode=='CBC':
            feedback_object=modes.CBC(iv)
        elif self.feedback_mode=='OFB':
            feedback_object=modes.OFB(iv)
        elif self.feedback_mode=='CFB':
            feedback_object=modes.CFB(iv)
        elif self.feedback_mode=='CTR':
            feedback_object=modes.CTR(iv)
        elif self.feedback_mode=='ECB':
            feedback_object=modes.ECB()

        if algorithm_object==None or feedback_object==None:
            raise ValueError("algorithm_object and feedback must not be None type")

        cipher = Cipher(algorithm_object,feedback_object, default_backend())
        encryptor = cipher.encryptor()
        padder=padding.PKCS7(cipher.algorithm.block_size).padder()

        # Open input file for reading and output file for writing
        with open(self.password, 'rb') as f_in, open(self.output_file, 'wb') as f_out:
            while True:
                # Read a chunk of the input file into the plaintext variable
                plaintext = f_in.read(16)

                if not plaintext:
                    break
                padded_data=padder.update(plaintext)
                ciphertext = encryptor.update(padded_data)
                f_out.write(ciphertext)
                # Write the ciphertext in the output file
            padded_data=padder.finalize()
            ciphertext=encryptor.update(padded_data)+encryptor.finalize()
            print(f'[{self.get_datetime()}] Finalize Ciphertext:{ciphertext}')
            f_out.write(ciphertext)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Encrypt a file using AES encryption")
    parser.add_argument("--folder",required=True,help="Folder if data are in folder")
    parser.add_argument("--key",required=True,help="path to the key file(read/write)")
    parser.add_argument("--input",required=True,help="path to the input file")
    parser.add_argument("--output",required=True,help="path to the output file")
    parser.add_argument("--config",required=True,help="Config file for deciphering")
    parser.add_argument("--feedback",required=True,help="Encryption feedback information",type=str,choices=["CBC","OFB","CFB","CTR","ECB"],default="CBC")
    parser.add_argument("--algorithm",required=True,help="Algorithm for dicipher(AES/DES)",type=str,choices=["AES","DES"],default="AES")
    
    parser.add_argument("--bmp",help="In case you want to read a bmp",action='store_true')
    
    encryptor=Encryptor(parser)
    encryptor.encrypt_password()