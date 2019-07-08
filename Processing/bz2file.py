def load_bz2file(file, dims):
    import bz2
    import numpy as np
    data = bz2.BZ2File(file,'rb').read()
    im = np.frombuffer(data,dtype='int16')
    return im.reshape(dims[-1::-1])