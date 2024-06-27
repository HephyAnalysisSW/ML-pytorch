from math import pi

#defaults = {
#    "topPad":True,
#    "bottomPad":True,
#}

plot_options = {

    "TT2l_EFT_delphes": {
    "tr_ttbar_pt": { 
        "binning":[16, 0, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.5, 1.5],
            },
    },
}

#for m in plot_options.keys():
#    for v in  plot_options[m].keys():
#        for k,val in defaults.items():
#            if k not in plot_options[m][v]:
#                plot_options[m][v][k]=val
