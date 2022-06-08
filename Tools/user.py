import os
if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
