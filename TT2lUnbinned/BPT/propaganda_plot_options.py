from math import pi

#defaults = {
#    "topPad":True,
#    "bottomPad":True,
#}

plot_options = {

    "analytic_2D": {
    "x": { 
        "binning":[32, -pi, pi], 
        "tex":"x",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.78, 1.22],
        #"topPad":False
        #"shape_y_range":[0, 1.4],
            },
    },

    "TTLep_bTagSys_paper": {
    "tr_ttbar_pt": { 
        "binning":[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.88, 1.12],
            },
    "tr_ttbar_mass": { 
        "binning":[750, 950, 1150, 1350, 1550, 1750, 1950, 2150, 2350, 2550, 2750, 2950, 3150, 3550], 
        "tex":"M(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.88, 1.12],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.88, 1.12],
            },
    },

    "delphes_TTLep_DY_red": {
    "tr_ttbar_pt": { 
        "binning":[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0,2],
            },
    "tr_ttbar_mass": { 
        "binning":[750, 950, 1150, 1350, 1550, 1750, 1950, 2150, 2350, 2550, 2750, 2950, 3150, 3550], 
        "tex":"M(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0,2],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.88, 1.12],
            },
    "tr_cos_phi_lab": { 
        "binning":[20, -1, 1], 
        "tex":"cos(#phi_{lab})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0,2],
            },
    "tr_abs_delta_phi_ll_lab": { 
        "binning":[20, 0, pi], 
        "tex":"|#Delta(#phi(ll))",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.9981, 1.0019],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0,2],
            },
    },


    "TTLep_scale_2D_paper":{
    "tr_ttbar_pt": { 
        "binning":[16, 0, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_ttbar_mass": { 
        "binning":[15, 750, 3750], 
        "tex":"M(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_ttbar_dEta": { 
        "binning":[16, -4.2, 4.2], 
        "tex":"#Delta#eta(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    "recoLep_dAbsEta": { 
        "binning":[13, -2.6, 2.6], 
        "tex":"#Delta|#eta|(l#bar{l})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.71, 1.29],
            },
    "recoLep_dEta": { 
        "binning":[16, -3.2, 3.2], 
        "tex":"#Delta#eta(l#bar{l})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_abs_delta_phi_ll_lab": { 
        "binning":[16, 0, pi], 
        "tex":"#Delta|#phi|(l#bar{l})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    },


    "TTLep_JERC_ptt_linear_paper": {
    "tr_ttbar_pt": { 
        "binning":[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.87, 1.13],
            },
    },

    "TTLep_JERC_linear_paper": {
    "tr_ttbar_pt": { 
        "binning":[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.87, 1.13],
            },
    "tr_ttbar_mass": { 
        "binning":[15, 750, 3750], 
        "tex":"M(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    "tr_ttbar_dEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.71, 1.29],
            },
    },

    "TTLep_leptonSF": {
    "tr_ttbar_pt": { 
        "binning":[16, 0, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.975, 1.025],
            },
    "tr_ttbar_mass": { 
        "binning":[15, 750, 3750], 
        "tex":"M(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.975, 1.025],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.975, 1.025],
            },
    "tr_ttbar_dEta": { 
        "binning":[26, -2.6, 2.6], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.975, 1.025],
            },
    "recoLep01_pt": { 
        "binning":[15, 0, 540], 
        "tex":"p_{T}(ll)",
        "legendCoordinates":[0.25,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.975, 1.025],
            },
    },


    "TTLep_PDF_paper": {
    "tr_ttbar_pt": { 
        "binning":[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800], 
        "tex":"p_{T}(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":True,
        "shape_y_range":[0.998, 1.002],
            },
    "tr_ttbar_mass": { 
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "binning":[15, 750, 3750], 
        "tex":"M(t#bar{t})",
        "logY":True,
        "shape_y_range":[0.9981, 1.0019],
            },
    "tr_ttbar_dEta": { 
        "binning":[16, -4.2, 4.2], 
        "tex":"#Delta#eta(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.9981, 1.0019],
            },
    "tr_ttbar_dAbsEta": { 
        "binning":[16, -3.2, 3.2], 
        "tex":"#Delta|#eta|(t#bar{t})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.9981, 1.0019],
            },
    "tr_cos_phi_lab": { 
        "binning":[20, -1, 1], 
        "tex":"cos(#phi_{lab})",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.9981, 1.0019],
            },
    "tr_abs_delta_phi_ll_lab": { 
        "binning":[20, 0, pi], 
        "tex":"|#Delta(#phi(ll))",
        "legendCoordinates":[0.65,0.6,0.9,0.87],
        "logY":False,
        "shape_y_range":[0.9981, 1.0019],
            },
    },

    "delphes_TTLep_MG_vs_Pow":{
        #"tr_ttbar_pt": {
        #    "binning":[16, 0, 800],
        #    "tex":"p_{T}(t#bar{t})",
        #    "legendCoordinates":[0.65,0.6,0.9,0.87],
        #    "logY":True,
        #    "shape_y_range":[0.905, 1.095],
        #        },
        "tr_ttbar_mass": {
            "binning":[10, 750, 3750],
            "tex":"M(t#bar{t})",
            "legendCoordinates":[0.65,0.6,0.9,0.87],
            "logY":True,
            "shape_y_range":[0,3.1],
                },
        "tr_ttbar_dAbsEta": {
            "binning":[13, -2.6, 2.6],
            "tex":"#Delta|#eta|(t#bar{t})",
            "legendCoordinates":[0.65,0.6,0.9,0.87],
            "logY":False,
            "shape_y_range":[0,3.1],
                },
        "tr_cos_phi_lab": { 
            "binning":[10, -1, 1], 
            "tex":"cos(#phi_{lab})",
            "legendCoordinates":[0.65,0.6,0.9,0.87],
            "logY":False,
            "shape_y_range":[0,3.1],
                },
        "tr_abs_delta_phi_ll_lab": { 
            "binning":[15, 0, pi], 
            "tex":"|#Delta(#phi(ll))",
            "legendCoordinates":[0.65,0.6,0.9,0.87],
            "logY":False,
            "shape_y_range":[0,3.1],
                },
            },
}

#for m in plot_options.keys():
#    for v in  plot_options[m].keys():
#        for k,val in defaults.items():
#            if k not in plot_options[m][v]:
#                plot_options[m][v][k]=val
