import sympy

x = sympy.symbols("x")
print(sympy.integrate((x**3*sympy.cos(x/2)+1/2)*sympy.sqrt(4-x**2), (x,-2,2)))