import scipy.misc
import numpy as np
SIMSUN = np.load('SIMSUN.npy')
simhei = np.load('simhei.npy')
simkai = np.load('simkai.npy')
hanyiheiqi = np.load('hanyiheiqi.npy')
hanyiluobo = np.load('hanyiluobo.npy')
huawenxinwei = np.load('huawenxinwei.npy')

scipy.misc.imsave('de.png', np.concatenate([SIMSUN[0], simhei[0], simkai[0], hanyiheiqi[0], hanyiluobo[0], huawenxinwei[0]], axis=1))
