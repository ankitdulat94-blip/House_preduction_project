import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data={
    "area":[1000,1200,1500,1800,2000,2200,2400,2800,3000,3500],
    "bedrooms":[2,2,3,3,3,4,4,4,5,5],
    "bathrooms":[1,2,2,2,3,3,3,4,4,5],
    "age":[10,8,5,7,6,4,3,5,2,1],
    "price":[40,48,60,70,75,90,100,115,130,150]
}
df=pd.DataFrame(data)
X=df[["area","bedrooms","bathrooms","age"]]
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,
test_size=0.2,
random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=Sequential([
    Dense(64,activation="relu",input_shape=(4,)),
    Dense(32,activation="relu"),
    Dense(16,activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)
history=model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=2,
    validation_split=0.2,
    verbose=1
)
loss,mae=model.evaluate(X_test,y_test)
print("Test MAE:",mae)

new_house=[[1600,3,2,4]]
new_house_scaled=scaler.transform(new_house)
predication=model.predict(new_house_scaled)
print("Predicted Price:",predication[0][0])
model.save("house_price_model.keras")