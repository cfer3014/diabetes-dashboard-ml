# ==============================
# app.py - Dashboard Diabetes Interactivo Nivel TOP
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Dashboard Diabetes Interactivo 🏥", layout="wide")

# ==============================
# 1. CARGAR DATOS
# ==============================
df_orig = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_orig[cols] = df_orig[cols].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df_orig[cols] = imputer.fit_transform(df_orig[cols])

X = df_orig.drop("Outcome", axis=1)
y = df_orig["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 2. ENTRENAR MODELOS
# ==============================
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = round((pred == y_test).mean(), 4)
    results[name] = acc

results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Accuracy"])

# Guardar modelo Random Forest y scaler para predicción TOP
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Random Forest métricas
y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()

# Clustering y PCA
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_orig['Cluster'] = clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
cluster_summary = df_orig.groupby('Cluster').mean()

# ==============================
# 3. SIDEBAR FILTROS
# ==============================
st.sidebar.header("Filtros de pacientes")
df_dynamic = df_orig.copy()

age_filter = st.sidebar.slider("Edad", int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max()), 
                               (int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max())))
bmi_filter = st.sidebar.slider("IMC (BMI)", float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max()), 
                               (float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max())))
glucose_filter = st.sidebar.slider("Glucosa", int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max()), 
                                   (int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max())))
cluster_filter = st.sidebar.multiselect("Cluster", options=sorted(df_dynamic['Cluster'].unique()), 
                                        default=sorted(df_dynamic['Cluster'].unique()))

df_filtered = df_dynamic[
    (df_dynamic['Age'] >= age_filter[0]) & (df_dynamic['Age'] <= age_filter[1]) &
    (df_dynamic['BMI'] >= bmi_filter[0]) & (df_dynamic['BMI'] <= bmi_filter[1]) &
    (df_dynamic['Glucose'] >= glucose_filter[0]) & (df_dynamic['Glucose'] <= glucose_filter[1]) &
    (df_dynamic['Cluster'].isin(cluster_filter))
]

# ==============================
# 4. DASHBOARD TABS
# ==============================
tabs = st.tabs(["Datos & EDA", "Distribución & Outliers", "Correlación",
                "Modelos", "Random Forest", "Clustering", "Predicción Nivel TOP"])

# ---- TAB 1: Datos ----
with tabs[0]:
    st.header("Datos filtrados")
    st.dataframe(df_filtered)
    st.write(f"Dimensiones: {df_filtered.shape}")
    st.write(df_filtered.describe())

# ---- TAB 2: Distribución & Outliers ----
with tabs[1]:
    st.header("Distribución de pacientes")
    fig = px.histogram(df_filtered, x='Outcome', color='Outcome', text_auto=True,
                       labels={'Outcome':'Diabetes'}, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    df_long = df_filtered.melt(var_name="Variable", value_name="Valor")
    fig = px.histogram(df_long, x="Valor", color="Variable", facet_col="Variable",
                       facet_col_wrap=4, nbins=15, color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

    st.header("Valores atípicos")
    fig = px.box(df_filtered, y=df_filtered.columns[:-2], points="all", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: Correlación ----
with tabs[2]:
    st.header("Matriz de correlación")
    corr = df_filtered.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu")
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 4: Comparación Modelos ----
with tabs[3]:
    st.header("Comparación de modelos")
    st.dataframe(results_df)
    fig = px.bar(results_df, x="Modelo", y="Accuracy", text="Accuracy", color="Accuracy",
                 color_continuous_scale="Viridis", range_y=[0.6,1])
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 5: Random Forest ----
with tabs[4]:
    st.header("Random Forest - Matriz de Confusión")
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicho", y="Real"))
    st.plotly_chart(fig, use_container_width=True)

    st.header("Curva ROC")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.2f}', line=dict(color='darkorange', width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="Curva ROC")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Importancia de variables")
    fig = px.bar(importances, x=importances.values, y=importances.index, orientation='h',
                 color=importances.values, color_continuous_scale='Teal', text=importances.values)
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 6: Clustering ----
with tabs[5]:
    st.header("Clusters de pacientes")
    fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=df_dynamic['Cluster'].astype(str),
                     labels={'x':'PCA1', 'y':'PCA2', 'color':'Cluster'}, size_max=10)
    st.plotly_chart(fig, use_container_width=True)
    st.header("Resumen por cluster")
    st.dataframe(cluster_summary)

# ---- TAB 7: Predicción Nivel TOP ----
with tabs[6]:
    st.sidebar.header("Predicción de nuevo paciente")
    preg = st.sidebar.slider("Embarazos", 0, 20, 1)
    gluc = st.sidebar.slider("Glucosa", 0, 200, 120)
    bp = st.sidebar.slider("Presión sanguínea", 0, 140, 70)
    skin = st.sidebar.slider("Grosor piel", 0, 100, 20)
    ins = st.sidebar.slider("Insulina", 0, 900, 80)
    bmi = st.sidebar.slider("IMC (BMI)", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Edad", 10, 100, 40)

    # Escalar datos nuevo paciente
    new_patient_scaled = scaler.transform([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
    pred_prob = rf_model.predict_proba(new_patient_scaled)[0][1]
    pred_label = int(pred_prob > 0.5)
    cluster_pred = kmeans.predict(new_patient_scaled)[0]

    st.header("Predicción en tiempo real para nuevo paciente")
    st.write(f"✅ Probabilidad de diabetes: **{pred_prob*100:.1f}%**")
    if pred_label == 1:
        st.error("⚠️ Alto riesgo de diabetes")
    else:
        st.success("✅ Bajo riesgo de diabetes")
    st.write(f"Cluster asignado: **{cluster_pred}**")

    # Historial
    if "historial_predicciones" not in st.session_state:
        st.session_state.historial_predicciones = pd.DataFrame(columns=[
            "Edad", "IMC", "Glucosa", "Presion", "Outcome Predicho",
            "Probabilidad", "Cluster"
        ])

    nueva_pred = {
        "Edad": age,
        "IMC": bmi,
        "Glucosa": gluc,
        "Presion": bp,
        "Outcome Predicho": "Alto riesgo" if pred_label==1 else "Bajo riesgo",
        "Probabilidad": round(pred_prob*100,1),
        "Cluster": cluster_pred
    }

    st.session_state.historial_predicciones = pd.concat([
        st.session_state.historial_predicciones, 
        pd.DataFrame([nueva_pred])
    ], ignore_index=True)

    st.subheader("📊 Historial de predicciones")
    st.dataframe(st.session_state.historial_predicciones)

    # Gráfico comparativo
    st.subheader("Comparación con pacientes existentes")
    df_plot = df_filtered.copy()
    df_plot['NuevoPaciente'] = 0
    new_patient = pd.DataFrame([{
        'Pregnancies': preg,
        'Glucose': gluc,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': ins,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Outcome': pred_label,
        'Cluster': cluster_pred,
        'NuevoPaciente': 1
    }])
    df_plot = pd.concat([df_plot, new_patient], ignore_index=True)
    fig = px.scatter(df_plot, x='Glucose', y='BMI', color='Cluster', 
                     symbol='NuevoPaciente', size='NuevoPaciente',
                     labels={'NuevoPaciente':'Paciente nuevo'},
                     hover_data=['Age','Pregnancies','Outcome'])
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

    # Exportación PDF y Excel
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generar PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Reporte de Predicción de Diabetes", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            for key, val in nueva_pred.items():
                pdf.cell(0, 8, f"{key}: {val}", ln=True)
            pdf.ln(5)
            plt.figure(figsize=(6,4))
            for cluster_id in sorted(df_plot['Cluster'].unique()):
                sub_df = df_plot[df_plot['Cluster'] == cluster_id]
                plt.scatter(sub_df['Glucose'], sub_df['BMI'],
                            s=sub_df['NuevoPaciente'].replace({0:40, 1:80}),
                            label=f"Cluster {cluster_id}")
            plt.xlabel("Glucosa")
            plt.ylabel("BMI")
            plt.title("Paciente vs Historial / Dataset")
            plt.legend()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name, format="PNG")
                tmpfile_path = tmpfile.name
            plt.close()
            pdf.image(tmpfile_path, x=10, y=None, w=180)
            os.remove(tmpfile_path)
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button("💾 Descargar PDF", pdf_bytes, "reporte_diabetes.pdf", "application/pdf")

with col2:
    if st.button("📊 Exportar Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja 1: Pacientes filtrados
            df_filtered.to_excel(writer, index=False, sheet_name="Pacientes Filtrados")
            # Hoja 2: Nuevo paciente
            pd.DataFrame([nueva_pred]).to_excel(writer, index=False, sheet_name="Nuevo Paciente")
            # Hoja 3: Resumen por Cluster
            cluster_summary.to_excel(writer, sheet_name="Resumen por Cluster")
            # Hoja 4: Historial de predicciones
            st.session_state.historial_predicciones.to_excel(writer, index=False, sheet_name="Historial")
        output.seek(0)
        st.download_button(
            "💾 Descargar Excel", 
            output, 
            "dashboard_diabetes.xlsx", 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )