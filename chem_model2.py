import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use('default')
plt.rc('font', family='Cambria Math')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
plt.rcParams['figure.figsize'] = (8, 5)
def consecutive_reaction(t, y, k1, k2):
    """Returns the derivatives of [A], [B], and [C] for a consecutive reaction A -> B -> C."""
    A, B, C = y
    dAdt = -k1 * A
    dBdt = k1 * A - k2 * B
    dCdt = k2 * B
    return [dAdt, dBdt, dCdt]

#Parameters
k1 = 0.5
k2 = 0.3
A0 = 1.0
B0 = 0.0
C0 = 0.0
t_span = (0, 10)

#Solve ODE
solution_consec = solve_ivp(
    fun=lambda t, y: consecutive_reaction(t, y, k1, k2),
    t_span=t_span,
    y0=[A0, B0, C0],
    t_eval=np.linspace(t_span[0], t_span[1], 200)
)

t_vals_consec = solution_consec.t
A_vals_consec = solution_consec.y[0]
B_vals_consec = solution_consec.y[1]
C_vals_consec = solution_consec.y[2]

#Plot
plt.figure()
plt.plot(t_vals_consec, A_vals_consec, label='[A](t)')
plt.plot(t_vals_consec, B_vals_consec, label='[B](t)')
plt.plot(t_vals_consec, C_vals_consec, label='[C](t)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Consecutive Reaction: A → B → C')
plt.legend()
plt.show()
