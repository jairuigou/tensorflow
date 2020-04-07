import os
import numpy as np

class DataManager():
    filepath = ''
    def __init__(self,filepath):
        self.filepath = filepath
    def load_data(self):
        if not os.path.exists(self.filepath):
            raise Exception("file not exists")
        try:
            file = open(self.filepath)
        except:
            raise Exception("open file error")
        else:
            rows,cols = self.get_dimension(file)
            totaldata = np.zeros((rows-1,cols))
            for row,line in enumerate(file.readlines()):
                numstr = ''
                col = 0
                for i in line:
                    if (i>='0' and i<='9') or i=='.' or i=='+' or i=='-':
                        numstr += i
                    else:
                        totaldata[row][col] = float(numstr)
                        col += 1
                        numstr = ''
            
            xdata = totaldata[:,:-1]
            ydata = totaldata[:,1]
            file.close()
            return xdata,ydata
    def get_dimension(self,file):
        linecount = 0
        for i,line in enumerate(file.readlines()):
            linecount += 1

        file.seek(0, os.SEEK_SET)

        line = file.readline()

        numcount = 0
        for i in line:
            if (i<'0' or i>'9') and i != '.' and i != '-' and i!='+':
                numcount += 1

        return linecount,numcount
                
