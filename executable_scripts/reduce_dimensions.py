#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:06:16 2018

@author: miri-o
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:16:13 2018

@author: miri-o
"""

import sys, getopt
sys.path.insert(0, "/media/miri-o/Documents")

from miris_tools.dimensional_reduction import dimensional_reduction

def main(argv):
   inputfile = ''
   outputfile = ''
   n=2
   method = 'PCA'
   try:
      opts, args = getopt.getopt(argv,"hi:o:n:m:",["ifile=","ofile=","n=","method="])
   except getopt.GetoptError:
      print('reduce_dimensions.py -i <inputfile> -o <outputfile> -n <n_dimensions> -m <method>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('reduce_dimensions.py -i <inputfile> -o <outputfile> -n <n_dimensions> -m <method>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-m", "--method"):
         method = arg
      elif opt in ("-n", "--n-gram"):
         n = int(arg)
          
   print('Input file is: {}, n = {}, method = {}'.format(inputfile, n, method))
   
   dimensional_reduction(inputfile, outputfile, n, method)
   
if __name__ == "__main__":
   main(sys.argv[1:])   