import os

training_data_dir      = "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/"

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    results_directory      = "/groups/hephy/cms/robert.schoefbeck/NN/results/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
    model_directory= "/Users/robert.schoefbeckcern.ch/ML-pytorch/models"
    data_directory = "/Users/robert.schoefbeckcern.ch/ML-pytorch/data"
else:
    plot_directory = './plots/'
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    results_directory      = "/groups/hephy/cms/robert.schoefbeck/NN/results/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
