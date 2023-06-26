import os
import argparse
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class Decryptor:
    def __init__(self,parser):
        args=parser.parse_args()
        folder=args.folder

        with open(os.path.join(folder,args.encrypted_input),'rb') as reader:
            self.encrypted_data=reader.read()
        
        with open(os.path.join(folder,args.key),'rb') as reader:
            self.cryptography_key=reader.read()
        
        configurations=dict()
        with open(os.path.join(folder,args.encryption_configurations),'r') as reader:
            for line in reader:
                data=line.strip().split(':')
                configurations[data[0].strip()]=data[1].strip()

        self.output=os.path.join(folder,args.output)

        self.algorithm=configurations['Algorithm']
        self.feedback_mode=configurations['Feedback Mode']
        self.iv=bytes.fromhex(configurations['IV'])


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
    
    def get_algorithm_mode_bytes(self):
        if self.algorithm in self.iv_sizes:
            if self.feedback_mode in self.iv_sizes[self.algorithm]:
                return self.iv_sizes[self.algorithm][self.feedback_mode]
        return 0
    
    def decryption(self):
        algorithm_obj=None
        feedback_obj=None
        block_size=None

        if self.algorithm=="AES":
            algorithm_obj=algorithms.AES(self.cryptography_key)
            block_size=algorithms.AES.block_size
        elif self.algorithm=="DES":
            algorithm_obj=algorithms.TripleDES(self.cryptography_key)
            block_size=algorithms.TripleDES.block_size
        if self.feedback_mode=="CBC":
            feedback_obj=modes.CBC(self.iv)
        elif self.feedback_mode=="OFB":
            feedback_obj=modes.OFB(self.iv)
        elif self.feedback_mode=="CFB":
            feedback_obj=modes.CFB(self.iv)
        elif self.feedback_mode=="CTR":
            feedback_obj=modes.CTR(self.iv)
        elif self.feedback_mode=="ECB":
            feedback_obj=modes.ECB()

        if algorithm_obj==None or feedback_obj==None:
            raise ValueError("algorithm obj and feedback object should not be setted None")

        cipher=Cipher(algorithm_obj,feedback_obj,default_backend())
        decryptor=cipher.decryptor()
        decrypted_data=decryptor.update(self.encrypted_data)+decryptor.finalize()
        unpadder=padding.PKCS7(block_size).unpadder()

        try:
            unpadded_data=unpadder.update(decrypted_data)+unpadder.finalize()
            with open(self.output,'wb') as writer:
                writer.write(unpadded_data)
        except ValueError:
            print("Error in Decryption")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",required=True,help="Folder which the data are saved at",default="data")
    parser.add_argument("--encrypted_input",required=True)
    parser.add_argument("--key",required=True,help="File to cryptographic key")
    parser.add_argument("--encryption_configurations",required=True,help="File to encryption configurations")
    parser.add_argument("--output",required=True)

    decryptor=Decryptor(parser)
    decryptor.decryption()
