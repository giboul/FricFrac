from numpy import log10
# https://en.wikipedia.org/wiki/Seismic_moment
mu = 0.95e9
width = 2.5
A = 0.05 * width
D = (0.202 - 0.184) * width
M0 = mu*A*D
Mw = 2/3 * log10(M0) - 10.7
print(f"{M0 = }")
print(f"{Mw = }")