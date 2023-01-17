import os

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
elif os.environ['USER'] in ['oskar.rothbacher']:
    plot_directory         = "/groups/hephy/cms/oskar.rothbacher/www/pytorch/"
    model_directory        = "/groups/hephy/cms/oskar.rothbacher/NN/models/"
    data_directory         = "/groups/hephy/cms/oskar.rothbacher/NN/data/"
elif os.environ['USER'] in ['dietrich.liko']:
    plot_directory         = "/groups/hephy/cms/dietrich.liko/www/pytorch/"
    model_directory        = "/groups/hephy/cms/dietrich.liko/NN/models/"
    data_directory         = "/groups/hephy/cms/dietrich.liko/NN/data/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
    model_directory= "/Users/robert.schoefbeckcern.ch/ML-pytorch/models"
    data_directory = "/Users/robert.schoefbeckcern.ch/ML-pytorch/data"
elif os.environ['USER'] in ['lena.wild']:
    plot_directory = './plots/'
    #plot_directory         = "/groups/hephy/cms/lena.wild/www/pytorch/"
    model_directory        = "/groups/hephy/cms/lena.wild/NN/models/"
    data_directory         = "/groups/hephy/cms/lena.wild/NN/data/"
