import argparse
import torch
   
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.discount = None
        self.batch_size = 64
