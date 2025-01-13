import streamlit as st
import numpy as np

st.image("img/neuronas.jpg", use_container_width=True)

num_w = st.slider("numero de peso:", min_value=1, max_value=10, value=1, key="num_w")

st.title("Peso")
col = st.columns(num_w)
w = []
for i in range(num_w):
    with col[i]:
        st.write("w",i)
        w_entrada = st.number_input(f"Peso w{i}", key=f"w{i}")
        w.append(w_entrada)
st.write(f"{w}")



st.title("Entradas")
col_2 = st.columns(num_w)
x = []
for i in range(num_w):
    with col_2[i]:
        st.write("x",i)
        x_entrada = st.number_input(f"Entrada x{i}", key=f"x{i}")
        x.append(x_entrada)
st.write(f"{x}")

sesgo, funcion = st.columns(2)

with sesgo:
    st.subheader("Sesgo")
    b = st.number_input("Ingrese el valor del sesgo:", value=0.0, key="sesgo")

with funcion:
    st.subheader("Función de Activación")
    funcion_activacion = st.selectbox(
        "Seleccione la función de activación:",
        ["Sigmoide", "ReLu", "Tangente hiperbólica", "Binary Step"],
        key="funcion"
    )

if st.button("Calcular Salida", type="primary", key="calcular"):
    w = np.array(w)
    x = np.array(x)

    if len(w) != len(x):
        st.error("Error: Los vectores de pesos y entradas deben tener el mismo tamaño.")
    else:
        y = np.dot(w, x) + b

        if funcion_activacion == "Sigmoide":
            salida = 1 / (1 + np.exp(-y))
        elif funcion_activacion == "ReLu":
            salida = np.maximum(0, y)
        elif funcion_activacion == "Tangente hiperbólica":
            salida = np.tanh(y)
        elif funcion_activacion == "Binary Step":
            salida = np.where(y > 0, 1, 0)

        st.success("El cálculo se realizó con éxito.")
        st.metric(label="Salida de la neurona:", value=f"{salida:.4f}")