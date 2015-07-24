Al = plotAll(["Al_foil_3/Al_1x_1p_long_*", "Al_foil_3/Al_300x_250p_long_*"],
subArray=[bgsubtract(10, 1, 501, 0., 1.), bgsubtract(250, 300, 501, 0., 1.)],
normalizationArray=[1./10, 563./250])
hot = peak_sizes(Al[0], Al[1], [[23., 35.], [41., 48.5]])
cold = peak_sizes(Al[0], Al[3], [[23., 35.], [41., 48.5]])
hot[0]/hot[1]
cold[0]/cold[1]
