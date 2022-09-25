N_events = 5000
N_channels = 11

expected_signal_yield = 100
expected_bkg_yield    = 20

sigma = 0.05

signal_events       = getEvents( N_events, N_channels, expected_bkg_yield=expected_bkg_yield, expected_signal_yield=expected_signal_yield, sigma=sigma)

background_events   = getEvents( N_events, N_channels, expected_bkg_yield=expected_bkg_yield, expected_signal_yield=0, sigma=sigma) 
