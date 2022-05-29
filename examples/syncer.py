'''
Prerequisites:
# add this to ~/.bash_profile
export KRB5_CONFIG=/mnt/hephy/cms/Tools/krb5.conf
Do 
kinit -fp <CERNUSER>@CERN.CH
not more than 24hrs before syncing.
The code wraps output functions and intercepts filenames.
All you have to do in your script is
import Analysis.Tools.syncer
and the filenames written via
    TCanvas::Print
    pickle.dump
will be synced.
'''

import os, uuid
import subprocess 
# Logger
import logging
logger = logging.getLogger(__name__)

# module singleton to keep track of files
file_sync_storage = []

## Wrap TCanvas class Print function
import ROOT
class myTCanvas( ROOT.TCanvas ):
    # recall the argument
    def Print( self, *args):
        logger.debug( "Appending file %s", args[0] )
        file_sync_storage.append( args[0] )
        # call original Print method 
        super(myTCanvas, self).Print(*args)
# what could possibly go wrong.
ROOT.TCanvas = myTCanvas 

from matplotlib import pyplot as plt

_savefig = plt.savefig
def my_savefig( *args, **kwargs):
    if not os.path.exists(os.path.dirname( args[0] ) ):
        os.makedirs( os.path.dirname( args[0] ) )
    file_sync_storage.append( args[0] )
    _savefig(*args, **kwargs)

plt.savefig = my_savefig

# Wrap pickle dump
import pickle
# that's the old dump method
pickle._dump = pickle.dump
def syncer_pickle_dump( *args ):
    # second argument is file handle!
    if len(args)>1:
        file_sync_storage.append( args[1].name )
    else:
        logger.warning( "Pickle dump called with less than two arguments... shouldn't happen." )
    pickle._dump(*args)
# that's the new dump method
pickle.dump = syncer_pickle_dump

# What happens on exit 
def write_sync_files_txt(output_filename = 'file_sync_storage.txt'):
    # No logger here, since it is already unloaded!
    # write file_sync_storage.txt because we can't scp in the container with kerberos authentication
    
    # find php files and copy these also
    for filename in list(set(file_sync_storage)):
        dir_path = os.path.dirname(os.path.realpath(filename))
        phpfiles = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.php')]
        for f in phpfiles:
            if f not in file_sync_storage:
                file_sync_storage.append(f)
    n_files = 0
    if len(file_sync_storage)>0:
        with open( output_filename, 'w' ) as outfile:
            for filename in file_sync_storage:
                if 'www/' in filename:
                    # for rsync cmd with relative path
                    outfile.write('{filename}\n'.format(filename=os.path.expanduser(os.path.expandvars(filename)).replace('www/','www/./')))
                    n_files+=1
                else:
                    print(("Will not sync %s" % filename))
        print(("Analysis.Tools.syncer: Written %i files to %s for rsync." % (n_files, output_filename)))
    return n_files

def sync():

    global file_sync_storage

    if len(file_sync_storage)==0:
        print ("No files for syncing.")
        return

    filename = '/tmp/%s.txt'%uuid.uuid4()

    if write_sync_files_txt(filename)==0: return 

    cmd = "rsync -avR  `cat %s` ${CERN_USER}@lxplus.cern.ch:/eos/user/$(echo ${CERN_USER} | head -c 1)/${CERN_USER}/www/" % filename
    #print cmd
    output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    os.remove(filename)
    file_sync_storage = []
    return #output, error

import atexit
atexit.register( sync )
