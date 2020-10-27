import talos
import matplotlib.pyplot as plt

analyze_object = talos.Analyze('general_hyperparameter.csv')


analyze_object.plot_bars('d_k','val_max_absolute_error','d_v','ff_dim')

plt.show()