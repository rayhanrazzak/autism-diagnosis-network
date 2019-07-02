'''import csv'''
def createLabel():
    for i in range(10):
        return i



import pandas as pd



df = pd.read_csv('anat_labels.csv', converters = {'id':createLabel})
