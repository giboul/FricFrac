from numpy import log10
# https://en.wikipedia.org/wiki/Seismic_moment
mu = 0.95e9  # Shear module of PMMA
l = 2.5  # m  # width of the plate
thk = 0.03  # m  # thickness of the plate
A = thk * l  # m²  # Area of fracture
D = (0.1075 - 0.1)/100 * l
M0 = mu*A*D
Mw = 2/3 * log10(M0) - 6.07
print(f"{M0 = }")
print(f"{Mw = }")