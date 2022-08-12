import os
if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
