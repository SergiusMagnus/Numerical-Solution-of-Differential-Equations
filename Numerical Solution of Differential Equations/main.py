from assembly import solve_DAE

# Methods:
# BEM - Backward Euler Method

if __name__ == '__main__':
    solve_DAE(start=0., end=0.2, step=0.0005, method="BEM")
