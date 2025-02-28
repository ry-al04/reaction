import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use('default')
plt.rc('font', family='Cambria Math')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
plt.rcParams['figure.figsize'] = (8, 5)
def first_order_reaction(t, y, k):
    """Returns the derivative of [A] with respect to time for a first-order reaction."""
    A = y[0]
    dAdt = -k * A
    return [dAdt]
k = 0.3     # rate constant
A0 = 1.0    # initial concentration of A
t_span = (0, 10)  # time range for the simulation
solution_1st = solve_ivp(
    fun=lambda t, y: first_order_reaction(t, y, k),
    t_span=t_span,
    y0=[A0],
    t_eval=np.linspace(t_span[0], t_span[1], 100)
)

t_vals_1st = solution_1st.t
A_vals_1st = solution_1st.y[0]
plt.figure()
plt.plot(t_vals_1st, A_vals_1st, label='[A](t)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('First-Order Reaction: A → P')
plt.legend()
plt.show()

