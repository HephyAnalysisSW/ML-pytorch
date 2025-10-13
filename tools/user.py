import os

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    results_directory      = "/groups/hephy/cms/robert.schoefbeck/NN/results/"
    training_data_dir      = "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/"
if os.environ['USER'] in ['rschoefbeck']:
    plot_directory         = "/scratch/robert.schoefbeck/www/pytorch/"
    model_directory        = "/scratch/robert.schoefbeck/NN/results/models/"
    data_directory         = "/scratch/rschoefbeck/NN/data/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
    model_directory= "/Users/robert.schoefbeckcern.ch/ML-pytorch/models"
    data_directory = "/Users/robert.schoefbeckcern.ch/ML-pytorch/data"
elif os.environ['USER'] in ['lena.wild']:
    plot_directory = './plots/'
    #plot_directory         = "/groups/hephy/cms/lena.wild/www/pytorch/"
    model_directory        = "/groups/hephy/cms/lena.wild/NN/models/"
    data_directory         = "/groups/hephy/cms/lena.wild/NN/data/"
