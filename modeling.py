

def plot_and_eval(train,validate,test,target_var):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat[target_var])
    plt.title(target_var)
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat[target_var])), 4)
    print(target_var, f'-- RMSE: {rmse}')
    plt.show()