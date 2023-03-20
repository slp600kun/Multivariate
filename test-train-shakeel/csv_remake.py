import pandas as pd
import datetime
import glob

def main():
    df = pd.read.csv(
        "csv/"
        dtype=str
    )