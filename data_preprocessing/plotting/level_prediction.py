import matplotlib.pyplot as plt

def plot_level_prediction(dates, predicted_level,real_level, step_ahead, text):
    # Visualising the results
    plt.plot(dates[(step_ahead - 1):], real_level, color='green', label='Real discharge')
    # plt.plot(val_x[:,-1,-1], color= 'blue', label='Rain')
    plt.plot(dates[(step_ahead - 1):], predicted_level, color='red', label='Predicted discharge')
    # plt.plot(val_dates[memory*(step_ahead-1):],np.abs(x-y), color = 'red', label = 'Error')
    plt.text(0, 1, text, transform=plt.gca().transAxes)
    plt.title('Discharge Prediction')
    plt.xlabel('Time')
    plt.ylabel('Discharge (m^3 / s)')
    plt.legend()
    plt.show()