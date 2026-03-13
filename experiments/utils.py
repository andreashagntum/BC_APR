'''This script contains some auxiliary functions.
'''
import sys
import os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
