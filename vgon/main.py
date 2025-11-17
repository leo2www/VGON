import cvxpy

def test():
    x = cvxpy.Variable()
    prob = cvxpy.Problem(cvxpy.Minimize((x - 2) ** 2))
    prob.solve()
    print(f"Optimal value: {x.value}")