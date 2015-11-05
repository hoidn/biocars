# test script for transmission.py
get_ipython().magic(u'run ../transmission.py')
get_ipython().magic(u'run ../rayonix_process.py')
import mu

b = beam_spectrum('None')
energies = b[0]
As = mu.ElementData('As')
Ge = mu.ElementData('Ge')
ge_as_att = np.exp(-(Ge.mu(energies) + As.mu(energies)) * 8e-4)
i2 = i1 * ge_as_att
ca = correct_attenuation(b, {'As': 0.5, 'Ge': 0.5}, target=[b[0], i2])
et_ipython().magic(u'run biocars/transmission.py')
ca = correct_attenuation(b, {'As': 0.5, 'Ge': 0.5}, target=[b[0], i2], bg_sub = True)
plt.plot(*ca)
plt.plot(energies, i2)
plt.show()
# The two curves now plotted should overlay each other.
