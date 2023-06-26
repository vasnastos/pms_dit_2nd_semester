import os,argparse
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

class Encryptor:
    def __init__(self,parser):
        args=parser.parse_args()
        folder=args.folder
        self.input_file=os.path.join(folder,args.input) if folder else args.input
        self.key_file=os.path.join(folder,args.key) if folder else args.key
        self.output_file=os.path.join(folder,args.output) if folder else args.outpute

    def encrypt_file_with_padding(self):
        key=None
        with open(self.key_file,'rb') as f:
            key=f.read()
        
        if key==None:
            raise ValueError("Key can not be setted to None value")

        #setup cipher: AES in CBC mode, with a random IV and no padding
        iv=os.urandom(algorithms.AES.block_size//8)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), default_backend())
        encryptor = cipher.encryptor()
        padder=padding.PKCS7(algorithms.AES.block_size).padder()

        # Open input file for reading and output file for writing
        with open(self.input_file, 'rb') as f_in, open(self.output_file+"_with_padding", 'wb') as f_out:
            # Write the contents of the IV in the output file
            f_out.write(iv)

            while True:
                # Read a chunk of the input file into the plaintext variable
                plaintext = f_in.read(16)

                if not plaintext:
                    ciphertext = encryptor.update(padder.finalize())
                    # Write the final ciphertext in the output file
                    print(f'{ciphertext=}')
                    f_out.write(ciphertext)
                    break
                else:
                    ciphertext = encryptor.update(plaintext)
                    # Write the ciphertext in the output file
                    f_out.write(ciphertext)
    
    def encrypt_file_no_padding(self):
        key=None
        with open(self.key_file,'rb') as f:
            key=f.read()
        
        if key==None:
            raise ValueError("Key can not be setted to None value")

        #setup cipher: AES in CBC mode, with a random IV and no padding
        iv=os.urandom(algorithms.AES.block_size//8)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), default_backend())
        encryptor = cipher.encryptor()

        # Open input file for reading and output file for writing
        with open(self.input_file, 'rb') as f_in, open(self.output_file+"_without_padding", 'wb') as f_out:
            ciphertext=b""
            # Write the contents of the IV in the output file
            f_out.write(iv)

            while True:
                # Read a chunk of the input file into the plaintext variable
                plaintext = f_in.read(16)

                if not plaintext:
                    break

                plaintext_len=len(plaintext)

                # Truncate the plaintext to the nearest smaller multiple of the block size
                if plaintext_len % algorithms.AES.block_size != 0:
                    continue
                #     plaintext = plaintext[:-(plaintext_len % algorithms.AES.block_size)]
                # print(len(plaintext))
                # if len(plaintext)%algorithms.AES.block_size!=0:
                #     raise ValueError("Plaintext length must be a multiple of the block size")
                ciphertext+=encryptor.update(plaintext)
                f_out.write(ciphertext)
            ciphertext+=encryptor.finalize()
            f_out.write(ciphertext)
            print(f'{ciphertext=}')

    def encrypt_file(self,padding=False):
        if padding:
            self.encrypt_file_with_padding()
        else:
            self.encrypt_file_no_padding()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Encrypt a file using AES encryption")
    parser.add_argument("--folder",required=True,help="Folder if data are in folder")
    parser.add_argument("--key",required=True,help="path to the key file")
    parser.add_argument("--input",required=True,help="path to the input file")
    parser.add_argument("--output",required=True,help="path to the output file")
    parser.add_argument("--with_padding",help="Using padding in the encryption",action='store_true')
    
    encryptor=Encryptor(parser)
    encryptor.encrypt_file(padding=True)
    encryptor.encrypt_file(padding=False)