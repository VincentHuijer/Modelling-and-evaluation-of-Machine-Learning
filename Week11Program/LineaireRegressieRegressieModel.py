import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from the Excel file
df = pd.read_excel('Outdoordb.xlsx', sheet_name=0)

# Define the features and target variables
X = df[['ORDER_DETAIL_CODE', 'RETURN_REASON_CODE', 'order_details.ORDER_NUMBER', 'order_details.PRODUCT_NUMBER', 'order_details.QUANTITY', 'order_details.UNIT_COST', 'order_details.UNIT_PRICE', 'order_details.UNIT_SALE_PRICE']]
y = df['RETURN_QUANTITY']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the target variable using the model
y_pred = model.predict(X)

# Calculate the correlation coefficient between the predicted and actual values
corr_coef = y.corr(pd.Series(y_pred))

# Visualize the linear regression
sns.regplot(x=y_pred, y=y)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title(f'Linear Regression (Correlation Coefficient: {corr_coef:.2f})')
plt.show()
