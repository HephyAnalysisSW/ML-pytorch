#!/usr/bin/env python

import os

# Logging
import logging
logger = logging.getLogger(__name__)

import uproot
import awkward as ak
import numpy as np
import os

def chunk( tot, n_split, index):
    ''' Implements split of number into n_split chunks
        https://math.stackexchange.com/questions/2975936/split-a-number-into-n-numbers
        Return tuple (start, stop)
    '''
        
    d = tot // n_split
    r = tot % n_split

    #return [d+1]*r + [d]*(n_split-r)
    if index<r:
        return ( (d+1)*index, (d+1)*(index+1) )
    else:
        return ( (d+1)*r + d*(index-r), (d+1)*r + d*(index-r+1) )

from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):

    def __init__( self, 
            input_files,
            branches            = None, 
            #padding         = 0.,
            n_split             = 1, 
            splitting_strategy  = "files",
            tree_name           = "Events",
            selection           = None,
                ):
        '''
        DataGenerator for the keras training framework.
        branches: which branches to return 
        n_split:    Number of chunks the data should be split into, use -1 and splitting_stragey = 'files' to split per file.
        input_files:    Input files or directories.
        '''

        # input_files
        self.input_files = []
        for filename in input_files:
            if filename.endswith('.root'):
                self.input_files.append(filename)
            # step into directory
            elif os.path.isdir( filename ):
                for filename_ in os.listdir( filename ):
                    if filename_.endswith('.root'):
                        self.input_files.append(os.path.join( filename, filename_ ))
            else:
                raise RuntimeError( "Don't know what to do with %r" % filename )

        self.splitting_strategy = splitting_strategy
        if splitting_strategy.lower() not in ['files', 'events']:
            raise RuntimeError("'splitting_strategy' must be 'files' or 'events'")

        # split per file
        if splitting_strategy == "files" and n_split<0:
            n_split = len(self.input_files)

        # apply selection string
        self.selection = selection        

        # Into how many chunks we split
        self.n_split        = n_split
        if not n_split>0 or not type(n_split)==int:
            raise RuntimeError( "Need to split in positive integer. Got %r"%n )
            
        # variables to return 
        self.branches = branches 

        # recall the index we loaded
        self.index          = -1

        # name of the tree to be read
        self.tree_name = tree_name

    # interface to Keras
    def __len__( self ):
        return self.n_split

    def load( self, index = -1, small = None):

        if index>=0:
            n_split = self.n_split
        else:
            n_split = 1
            index   = 0

        # load the files (don't load the data)
        if self.splitting_strategy.lower() == 'files':
            filestart, filestop = chunk( len(self.input_files), n_split, index )
        else:
            filestart, filestop = 0, len(self.input_files)

        array = uproot.concatenate([f+':'+self.tree_name for f in self.input_files], self.branches)
        # apply selection, if any
        len_before = len(array)
        if self.selection is not None: 
            array = array[self.selection(array)]
            print ("Applying selection with efficiency %4.3f" % (len(array)/len_before) )

        if self.splitting_strategy.lower() == 'events':
            entry_start, entry_stop = chunk( len(array), n_split, index )
        else:
            entry_start, entry_stop = 0, len(array)

        if small is not None and small>0:
            entry_stop = min( entry_stop, entry_start+small )

        self.index = index

        if self.selection is not None: 
            self.data = array[entry_start:entry_stop]
        else:
            self.data = array[entry_start:entry_stop]

        return self.data

    def __getitem__(self, index):
        if index == self.index:
            return self.data
        else:
            return self.load( index )

    def scalar_branches( self, branches ):
    
        #d=[]
        #for b in branches: 
        #    print (b)
        #    d.append( self.data[b].to_list() )       

        #return np.array( d ).transpose()
        return np.array( [ self.data[b].to_list() for b in branches ] ).transpose()

    def vector_branch( self, branches, padding_target=50, padding_value=0.):
        if type(branches)==str:
            return np.array(self.data[branches].to_list())
        else:
            return np.array([ np.array(ak.fill_none(ak.pad_none(self.data[b].to_list(), target=padding_target, clip=True), value=padding_value)).transpose() for b in branches ]).transpose()

if __name__=='__main__':

    data = DataGenerator(
        input_files = ["/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/"],
            n_split = 1,
            splitting_strategy = "files",
            branches = ["genJet_pt", "genJet_eta", "nchh"], 
        ) 

