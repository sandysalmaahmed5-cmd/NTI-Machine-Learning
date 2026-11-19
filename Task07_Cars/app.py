import joblib
import pandas as pd
import streamlit as st

# Load everything the notebook saved
saved = joblib.load('model.joblib')
model = saved['model']
scaler = saved['scaler']
inputs = saved['inputs']
cat_cols = saved['cat_cols']
dummy_columns = saved['dummy_columns']

st.title('Car Price Predictor')
st.write('Enter the car details and click Predict.')

# One input widget for each column
values = {}
for col, info in inputs.items():
    if info[0] == 'choice':
        values[col] = st.selectbox(col, info[1])
    else:
        values[col] = st.number_input(col, min_value=info[1], max_value=info[2], value=info[3])

if st.button('Predict'):
    # Same preprocessing as the notebook: dummies -> same columns -> scaling
    row = pd.DataFrame([values])
    row = pd.get_dummies(row, columns=cat_cols)
    row = row.reindex(columns=dummy_columns, fill_value=0).astype(float)
    row = scaler.transform(row)
    pred = model.predict(row)[0]
    st.success(f'Predicted price: {pred:,.2f}')
