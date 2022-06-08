import os
if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
elif os.environ['USER'] in ['robert.schoefbeckcern.ch']:
    plot_directory = './plots/'
elif os.environ['USER'] in ['sridhar.busulu']:
    plot_directory         = "/groups/hephy/cms/sridhar.busulu/www/pytorch/"
