# coding: utf-8
get_ipython().magic(u'run biocars/rayonix_process.py')
get_ipython().magic(u'run -i plotAu.py')
peak_angles = [.445, .516, .739, .875]
# peaks normalized to 111 intensity
plt.plot(peak_angles, coldn, 'o-', label = 'Au 300x')
plt.plot(peak_angles, mediumn, 'o-', label = 'Au 10x')
plt.plot(peak_angles, hotn, 'o-', label = 'Au 1x')
plt.xlabel('Scattering angle (rad)')
plt.ylabel('Normalized Bragg peak intensity')
plt.legend()
plt.show()
# no normalization
plt.plot(peak_angles, cold, 'o-', label = 'Au 300x')
plt.plot(peak_angles, medium, 'o-', label = 'Au 10x')
plt.plot(peak_angles, hot, 'o-', label = 'Au 1x')
plt.xlabel('Scattering angle (rad)')
plt.ylabel('Bragg peak intensity (arb)')
plt.legend()
plt.show()
