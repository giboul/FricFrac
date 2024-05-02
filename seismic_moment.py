from numpy import log10
# https://en.wikipedia.org/wiki/Seismic_moment
mu = 0.95e9  # Shear module of PMMA
width = 2.5  # m  # width of the plate
thk = 0.05  # m  # thickness of the plate
A = thk * width  # m²  # Area of fracture
D = (6.48e-2 - 5.52e-2) * width
M0 = mu*A*D
Mw = 2/3 * log10(M0) - 10.7
print(f"{M0 = }")
print(f"{Mw = }")