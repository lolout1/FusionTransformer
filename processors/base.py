
from abc import ABC, abstractmethod
from typing import List
import numpy as np 
class Processor():
    def __init__(self, mode: str, duration:int):
        '''
        Input : 

        '''
        self.duration = 2
        self.mode = mode

    def _get_file_name(self, path:str)-> str:
        file_name = path.partition('.')[0].split('/')[-1]
        return file_name

    @abstractmethod
    def process_feature(self, in_path : str, out_path: str) :
        '''
        A function to process the features 
        Input: 

        Output: 

        '''
        pass
    
    @abstractmethod
    def process_label(self,  path_list: List[str]) -> np.ndarray:
        '''
        '''
        pass


    def set_patterns(self, activity_pattern, label_pattern ): 
        self.activity_pattern =  activity_pattern
        self.label_pattern = label_pattern
    
    
    

