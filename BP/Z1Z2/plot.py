import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """Zoom in on an inset plot and draw connecting lines
    ax:         The canvas returned by calling plt.subplots. For example: fig, ax = plt.subplots(1, 1)
    axins:      The canvas for the inset plot. For example: axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))
    zone_left:  The left endpoint of the x-coordinate of the zoomed region
    zone_right: The right endpoint of the x-coordinate of the zoomed region
    x:          X-axis labels
    y:          A list of all y-values
    linked:     The position to draw the connecting lines, {'bottom', 'top', 'left', 'right'}
    x_ratio:    X-axis scaling ratio
    y_ratio:    Y-axis scaling ratio
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)
    axins.tick_params(labelsize=8)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black", linewidth=1)

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)


if __name__ == '__main__':
    seed = 923
    nq, nl = 20, 400 
    fig, axes = plt.subplots(1, 1)

    # PLOT HISTORY
    vqe_energy = np.genfromtxt('result/data/vqe_nq%d_nl%d_seed%d_energy.csv' % (nq, nl, seed), delimiter=',') * nq # energy density is saved
    vqe_sa_energy = np.genfromtxt('result/data/vqe_nq%d_nl%d_seed%d_init1e-2_energy.csv' % (nq, nl, seed), delimiter=',')
    vgon_history = np.genfromtxt('result/data/vgon_nq%d_nl%d_seed%d_history.csv' % (nq, nl, seed), delimiter=',')
    vgon_min = vgon_history[:-1, 0]
    vgon_avg = vgon_history[:-1, 1]

    x = range(len(vqe_energy))
    axes.plot(x, vqe_energy, '--', label='VQE', markersize=1, color='#324675')
    axes.plot(x, vqe_sa_energy, '--', label='VQE-SA', markersize=1, color='#AB2A3C')
    axes.plot(x, vgon_avg, '--', label='VGON-Avg', markersize=1, color='#6FAE45')
    axes.plot(x, vgon_min, '--', label='VGON-Min', markersize=1, color='#660874')
    axes.plot(x, [-1]*len(x), '--', label='Ground Energy', markersize=2, color='gray')

    axins = axes.inset_axes((0.33, 0.13, 0.6, 0.42))
    axins.plot(x, vqe_energy, '--', markersize=1, color='#324675')
    axins.plot(x, vqe_sa_energy, '--', markersize=1, color='#AB2A3C')
    axins.plot(x, vgon_avg, '--', markersize=1, color='#6FAE45')
    axins.plot(x, vgon_min, '--', markersize=1, color='#660874')
    axins.plot(x, [-1]*len(x), '--', markersize=2, color='gray')
    axins.grid(color='#DFDFDF')

    y = [vgon_avg, vgon_min]
    zone_and_linked(axes, axins, 5, 50, x, y, 'top')

    axes.set_xlabel('Iteration')
    axes.set_ylabel('Energy')
    axes.grid(color='#DFDFDF')
    axes.legend()

    fig.savefig('result/figure/energy.pdf', bbox_inches='tight')

    # PLOT GRADIENTS
    vqe_grads = np.genfromtxt('result/data/vqe_nq%d_nl%d_seed%d_grads.csv' % (nq, nl, seed), delimiter=',')
    vqe_sa_grads = np.genfromtxt('result/data/vqe_nq%d_nl%d_seed%d_init1e-2_grads.csv' % (nq, nl, seed), delimiter=',')
    vgon_grads = np.genfromtxt('result/data/vgon_nq%d_nl%d_seed%d_grads.csv' % (nq, nl, seed), delimiter=',')
    
    vqe0 = pd.DataFrame(vqe_grads[0, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe0.values
    vqe0['Iteration'] = 0
    vqe1 = pd.DataFrame(vqe_grads[150, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe1.values
    vqe1['Iteration'] = 150
    vqe2 = pd.DataFrame(vqe_grads[-1, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe2.values
    vqe2['Iteration'] = 300
    vqe = pd.concat([vqe0, vqe1, vqe2])
    vqe[''] = 'VQE'
    print('Var(VQE-0)', np.var(vqe_grads[0, :]))
    print('Var(VQE-150)', np.var(vqe_grads[150, :]))
    print('Var(VQE-300)', np.var(vqe_grads[-1, :]))

    vqe_sa0 = pd.DataFrame(vqe_sa_grads[0, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe_sa0.values
    vqe_sa0['Iteration'] = 0
    vqe_sa1 = pd.DataFrame(vqe_sa_grads[150, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe_sa1.values
    vqe_sa1['Iteration'] = 150
    vqe_sa2 = pd.DataFrame(vqe_sa_grads[-1, :], columns=['Absolute Value of Gradient']).abs()
    v = vqe_sa2.values
    vqe_sa2['Iteration'] = 300
    vqe_sa = pd.concat([vqe_sa0, vqe_sa1, vqe_sa2])
    vqe_sa[''] = 'VQE-SA'
    print('Var(VQE-SA-0)', np.var(vqe_sa_grads[0, :]))
    print('Var(VQE-SA-150)', np.var(vqe_sa_grads[150, :]))
    print('Var(VQE-SA-300)', np.var(vqe_sa_grads[-1, :]))

    vgon0 = pd.DataFrame(vgon_grads[0, :], columns=['Absolute Value of Gradient']).abs()
    v = vgon0.values
    vgon0['Iteration'] = 0
    vgon1 = pd.DataFrame(vgon_grads[150, :], columns=['Absolute Value of Gradient']).abs()
    v = vgon1.values
    vgon1['Iteration'] = 150
    vgon2 = pd.DataFrame(vgon_grads[-1, :], columns=['Absolute Value of Gradient']).abs()
    v = vgon2.values
    vgon2['Iteration'] = 300
    vgon = pd.concat([vgon0, vgon1, vgon2])
    vgon[''] = 'VGON'
    print('Var(VGON-0)', np.var(vgon_grads[0, :]))
    print('Var(VGON-150)', np.var(vgon_grads[150, :]))
    print('Var(VGON-300)', np.var(vgon_grads[-1, :]))

    data = pd.concat([vqe, vqe_sa, vgon])

    fig, ax = plt.subplots()
    colors = ['#324675', '#AB2A3C', '#660874']
    ax = sns.stripplot(data=data, x='Iteration', y='Absolute Value of Gradient', hue='', jitter=0.32, edgecolor="none", size=1, dodge=True, palette=colors, legend=False)
    ax = sns.boxplot(data=data, x='Iteration', y='Absolute Value of Gradient', hue='', width=0.8, boxprops={'facecolor':'none', "zorder":8}, palette=colors, gap=0.15)
    ax.set_yscale('log')
    plt.grid(True)

    fig.savefig('result/figure/grads.pdf', bbox_inches='tight')

