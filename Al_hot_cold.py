#Al = plotAll(["Al_foil_3/Al_1x_1p_long_*", "Al_foil_3/Al_300x_250p_long_*"],
#subArray=[bgsubtract(10, 1, 501, 0., 1.), bgsubtract(250, 300, 501, 0., 1.)],
#normalizationArray=[1./10, 563./250])
#hot = peak_sizes(Al[0], Al[1], [[23., 35.], [41., 48.5]])
#cold = peak_sizes(Al[0], Al[3], [[23., 35.], [41., 48.5]])
#hot[0]/hot[1]
#cold[0]/cold[1]

estAl_300x = full_process("Al_foil_3/Al_300x*", 'Ag', dtheta = 1e-3, npulses =  250)
estAl_10x = full_process("Al_foil/Al_foil_10x_1p*", 'Ti', dtheta = 1e-3, npulses =  1)
estAl_1x = full_process("Al_foil_3/Al_1x*", 'None', dtheta = 1e-3, npulses =  10)
plt.plot(*estAl_300x(100), label = "Al 300x")
plt.plot(*estAl_10x(100), label = "Al 10x")
plt.plot(*estAl_1x(100), label = "Al 1x")
#plt.plot(*estAl_300x(0), label = "Al 300x un-deconvolved")
#plt.plot(*estAl_10x(0), label = "Al 10x un-deconvolved")
#plt.plot(*estAl_1x(0), label = "Al 1x un-deconvolved")
plt.legend()
plt.xlabel("Scattering angle (rad)")
plt.ylabel("Intensity (arb)")
plt.show()
medium = np.array(peak_sizes(estAl_10x(100)[0], estAl_10x(100)[1], [[.43, .46], [.505, .525], [.72, .75], [.85, .89]]))
hot = np.array(peak_sizes(estAl_1x(100)[0], estAl_1x(100)[1], [[.43, .46], [.505, .525], [.72, .75], [.85, .89]]))
cold = np.array(peak_sizes(estAl_300x(100)[0], estAl_300x(100)[1], [[.43, .46], [.505, .525], [.72, .75], [.85, .89]]))

hotn = hot/hot[0]
mediumn = medium/medium[0]
coldn = cold/cold[0]

