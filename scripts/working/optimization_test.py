from scipy.optimize import minimize

# objective function
def func(x):
    return x**2

# constraint equation
def cons(x):
    return -(x + 1)

# 制約式が非負になるようにする
cons = (
    {'type': 'ineq', 'fun': cons}
)

x = -10 # initial value

result = minimize(func, x0=x, constraints=cons, method='SLSQP')

print(result)