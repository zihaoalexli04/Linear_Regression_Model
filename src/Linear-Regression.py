import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def Clean_data():
    Boston_data = pd.read_csv("../Data/BostonHousing.csv")

    # Data Preparation
    # data separation
    y = Boston_data["zn"]
    X = Boston_data.drop(["zn"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # Model Building
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    LinearRegression()

    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    print(y_lr_train_pred, y_lr_test_pred)
    y_lr_test_pred

    # Evaluate model perfoemance
    y_lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    y_lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    print("LR MSE (Train):", y_lr_train_mse)
    print("LR R2 (Train):", lr_train_r2)
    print("LR MSE (Test):", y_lr_test_mse)
    print("LR r2 (Test):", lr_test_r2)

    lr_results = pd.DataFrame(
        ["Linear regression", y_lr_train_mse, lr_train_r2, y_lr_test_mse, lr_test_r2]
    ).transpose()
    lr_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]

    # Random Forest

    rf = RandomForestRegressor(max_depth=2, random_state=100)
    rf.fit(X_train, y_train)

    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    y_rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    y_rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)
    rf_results = pd.DataFrame(
        ["Linear regression", y_rf_train_mse, rf_train_r2, y_rf_test_mse, rf_test_r2]
    ).transpose()
    rf_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]
    # Model comparison
    df_models = pd.concat([lr_results, rf_results], axis=0)

    df_models.reset_index(drop = True)
    
    #Data visualization of prediction results
    
    plt.figure(figsize=(5,5))
    plt.scatter(x=y_train, y=y_lr_train_pred, alpha= 0.3)
    
    z = np.polyfit(y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)
    
    plt.plot(y_train, p(y_train), '#F8766D')
    plt.ylabel('Predict LogS')
    plt.xlabel('Experimental LogS')
    
    
        
    
