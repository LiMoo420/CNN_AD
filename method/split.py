'''
# This script is to split the original data and it's file folds 
# into 'test', 'train', 'val' categories for neural network classification.

# Author: Bo Yin[MC36455*] & Zihan Xue[MC36588*]
# Contact: mc36455@um.edu.mo For Mr.Bo Yin
#          mc36588@um.edu.mo For Ms.Zihan Xue
'''

#==============================================================================

import splitfolders

#==============================================================================

# Parameters Definition About split fileFolders into train/test/val folders
raw_data_path = 'raw_data'
processed_data_path = 'processed_data'
random_seed = 420
train_ratio, test_ratio, validation_ratio = (0.8, 0.1, 0.1)
split_ratio = [train_ratio, test_ratio, validation_ratio]

#==============================================================================
def split_folders(raw_path:str, pro_path:str, seed:int, ratio = []):
    '''
    # Split raw data filefolders
    #
    # Input
    # ====
    # `raw_path`: (str) The directory of original data.
    # `pro_path:str`: (str) The directory of data after splited.
    # `seed`: (int) Random seed
    # `ratio`: (list) Contain a list of split ratio
    '''
    splitfolders.ratio(
        input = raw_path, 
        output = pro_path, 
        seed = seed, ratio = (ratio[0], ratio[1], ratio[2])
    )
