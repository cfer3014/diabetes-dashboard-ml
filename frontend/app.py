# # ==============================
# # app.py - Dashboard Diabetes Interactivo Nivel TOP
# # ==============================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import joblib
# import io
# from fpdf import FPDF
# import matplotlib.pyplot as plt
# import tempfile
# import os

# st.set_page_config(page_title="Dashboard Diabetes Interactivo 🏥", layout="wide")

# st.markdown("""
# <h1 style='text-align: center; font-size:25px; color: #00BFA6;'>
# 🏥 Predicción Inteligente de Diabetes
# </h1>
# <p style='text-align: center; font-size:18px;'>
# Predicción • Segmentación • Análisis en tiempo real
# </p>
# """, unsafe_allow_html=True)

# st.markdown("""
# <style>

# /* Subheaders estilo premium */
# h2 {
#     font-size: 18px !important;
#     font-weight: 600;
#     color: #E5E7EB;
    
#     padding-left: 10px;
#     margin-top: 10px;
#     margin-bottom: 8px;
# }
# .kpi {
#     background-color: #1f2937;
#     padding: 10px;              /* antes 20px */
#     border-radius: 8px;
#     text-align: center;
#     color: white;
#     border-left: 4px solid #00BFA6;
# }

# .kpi h2 {
#     font-size: 22px;            /* antes grande */
#     margin: 0;
# }

# .kpi h3 {
#     font-size: 14px;
#     margin: 0;
#     color: #9CA3AF;
# }
# </style>
# """, unsafe_allow_html=True)

# # ==============================
# # 1. CARGAR DATOS
# # ==============================
# df_orig = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# df_orig[cols] = df_orig[cols].replace(0, np.nan)
# imputer = SimpleImputer(strategy='median')
# df_orig[cols] = imputer.fit_transform(df_orig[cols])

# X = df_orig.drop("Outcome", axis=1)
# y = df_orig["Outcome"]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# # ==============================
# # 2. ENTRENAR MODELOS
# # ==============================
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "KNN": KNeighborsClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
# }

# results = {}
# for name, m in models.items():
#     m.fit(X_train, y_train)
#     pred = m.predict(X_test)
#     acc = round((pred == y_test).mean(), 4)
#     results[name] = acc

# results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Precision"])

# # Guardar modelo Random Forest y scaler para predicción TOP
# rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
# rf_model.fit(X_train, y_train)
# joblib.dump(rf_model, "model.pkl")
# joblib.dump(scaler, "scaler.pkl")

# # Random Forest métricas
# y_pred = rf_model.predict(X_test)
# y_probs = rf_model.predict_proba(X_test)[:,1]
# fpr, tpr, _ = roc_curve(y_test, y_probs)
# roc_auc = auc(fpr, tpr)
# cm = confusion_matrix(y_test, y_pred)
# importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()

# # Clustering y PCA
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)
# df_orig['Cluster'] = clusters
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# cluster_summary = df_orig.groupby('Cluster').mean()

# # ==============================
# # 3. SIDEBAR FILTROS
# # ==============================
# st.sidebar.header("Filtros de pacientes")
# df_dynamic = df_orig.copy()

# age_filter = st.sidebar.slider("Edad", int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max()), 
#                                (int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max())))
# bmi_filter = st.sidebar.slider("IMC (BMI)", float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max()), 
#                                (float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max())))
# glucose_filter = st.sidebar.slider("Glucosa", int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max()), 
#                                    (int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max())))
# cluster_filter = st.sidebar.multiselect("Cluster", options=sorted(df_dynamic['Cluster'].unique()), 
#                                         default=sorted(df_dynamic['Cluster'].unique()))

# df_filtered = df_dynamic[
#     (df_dynamic['Age'] >= age_filter[0]) & (df_dynamic['Age'] <= age_filter[1]) &
#     (df_dynamic['BMI'] >= bmi_filter[0]) & (df_dynamic['BMI'] <= bmi_filter[1]) &
#     (df_dynamic['Glucose'] >= glucose_filter[0]) & (df_dynamic['Glucose'] <= glucose_filter[1]) &
#     (df_dynamic['Cluster'].isin(cluster_filter))
# ]


# st.header("📊 Indicadores Clave")

# col1, col2, col3, col4 = st.columns(4)

# col1.markdown(f"""
# <div class="kpi">
# <h3>Pacientes</h3>
# <h2>{len(df_filtered)}</h2>
# </div>
# """, unsafe_allow_html=True)

# col2.markdown(f"""
# <div class="kpi">
# <h3>Riesgo Promedio</h3>
# <h2>{df_filtered['Outcome'].mean()*100:.1f}%</h2>
# </div>
# """, unsafe_allow_html=True)

# col3.markdown(f"""
# <div class="kpi">
# <h3>Glucosa Promedio</h3>
# <h2>{df_filtered['Glucose'].mean():.1f}</h2>
# </div>
# """, unsafe_allow_html=True)

# col4.markdown(f"""
# <div class="kpi">
# <h3>Clusters</h3>
# <h2>{df_filtered['Cluster'].nunique()}</h2>
# </div>
# """, unsafe_allow_html=True)

# st.header("🧠 Insight automático")

# riesgo = df_filtered['Outcome'].mean()

# if riesgo > 0.6:
#          st.error("⚠️ Alto riesgo general en la población analizada")
# elif riesgo > 0.3:
#           st.warning("⚠️ Riesgo moderado detectado en la población analizada")
# else:
#           st.success("✅ Bajo riesgo general en la población")


# # ==============================
# # 4. DASHBOARD TABS
# # ==============================
# tabs = st.tabs(["Datos & EDA", "Distribución & Outliers", "Correlación",
#                 "Modelos", "Random Forest", "Clustering", "Predicción Nivel TOP"])

# # ---- TAB 1: Datos ----
# with tabs[0]:
#     st.header("Datos filtrados")
#     st.dataframe(df_filtered)
#     st.write(f"Dimensiones: {df_filtered.shape}")
#     st.write(df_filtered.describe())

# # ---- TAB 2: Distribución & Outliers ----
# with tabs[1]:
#     st.header("Distribución de pacientes")
#     fig = px.histogram(df_filtered, x='Outcome', color='Outcome', text_auto=True,
#                        labels={'Outcome':'Diabetes'}, color_discrete_sequence=px.colors.qualitative.Set2)
#     st.plotly_chart(fig, use_container_width=True)

#     df_long = df_filtered.melt(var_name="Variable", value_name="Valor")
#     fig = px.histogram(df_long, x="Valor", color="Variable", facet_col="Variable",
#                        facet_col_wrap=4, nbins=15, color_discrete_sequence=px.colors.qualitative.Set3)
#     st.plotly_chart(fig, use_container_width=True)

#     st.header("Valores atípicos")
#     fig = px.box(df_filtered, y=df_filtered.columns[:-2], points="all", color_discrete_sequence=px.colors.qualitative.Set3)
#     st.plotly_chart(fig, use_container_width=True)

# # ---- TAB 3: Correlación ----
# with tabs[2]:
#     st.header("Matriz de correlación")
#     corr = df_filtered.corr()
#     fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu")
#     st.plotly_chart(fig, use_container_width=True)

# # ---- TAB 4: Comparación Modelos ----
# with tabs[3]:
#     st.header("Comparación de modelos")
#     st.dataframe(results_df)
#     fig = px.bar(results_df, x="Modelo", y="Precision", text="Precision", color="Precision",
#                  color_continuous_scale="Viridis", range_y=[0.6,1])
#     st.plotly_chart(fig, use_container_width=True)

# # ---- TAB 5: Random Forest ----
# with tabs[4]:
#     st.header("Random Forest - Matriz de Confusión")
#     fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
#                     labels=dict(x="Predicho", y="Real"))
#     st.plotly_chart(fig, use_container_width=True)

#     st.header("Curva ROC - (Receiver Operating Characteristic)")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.2f}', line=dict(color='darkorange', width=3)))
#     fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
#     fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",title="")
#     st.plotly_chart(fig, use_container_width=True)

#     st.header("Importancia de variables")
#     fig = px.bar(importances, x=importances.values, y=importances.index, orientation='h',
#                  color=importances.values, color_continuous_scale='Teal', text=importances.values)
#     st.plotly_chart(fig, use_container_width=True)

# # ---- TAB 6: Clustering ----
# with tabs[5]:
#     st.header("Clusters de pacientes")
#     fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=df_dynamic['Cluster'].astype(str),
#                      labels={'x':'PCA1', 'y':'PCA2', 'color':'Cluster'}, size_max=10)
#     st.plotly_chart(fig, use_container_width=True)
#     st.header("Resumen por cluster")
#     st.dataframe(cluster_summary)

# # ---- TAB 7: Predicción Nivel TOP ----
# with tabs[6]:
#     st.sidebar.header("Predicción de nuevo paciente")
#     preg = st.sidebar.slider("Embarazos", 0, 20, 1)
#     gluc = st.sidebar.slider("Glucosa", 0, 200, 120)
#     bp = st.sidebar.slider("Presión sanguínea", 0, 140, 70)
#     skin = st.sidebar.slider("Grosor piel", 0, 100, 20)
#     ins = st.sidebar.slider("Insulina", 0, 900, 80)
#     bmi = st.sidebar.slider("IMC (BMI)", 0.0, 70.0, 25.0)
#     dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
#     age = st.sidebar.slider("Edad", 10, 100, 40)
    
    
#     # Escalar datos nuevo paciente
#     new_patient_scaled = scaler.transform([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
#     pred_prob = rf_model.predict_proba(new_patient_scaled)[0][1]
#     pred_label = int(pred_prob > 0.5)
#     cluster_pred = kmeans.predict(new_patient_scaled)[0]



#     st.header("🧾 Predicción en tiempo real para nuevo paciente")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Probabilidad", f"{pred_prob*100:.1f}%")
#     col2.metric("Cluster", cluster_pred)
#     col3.metric("Riesgo", "Alto" if pred_label==1 else "Bajo")

#     if pred_label == 1:
#         st.error("⚠️ El paciente presenta ALTO riesgo de diabetes")
#     else:
#         st.success("✅ El paciente presenta BAJO riesgo de diabetes")
      

#     #st.header("Predicción en tiempo real para nuevo paciente")
#     # st.write(f"✅ Probabilidad de diabetes: **{pred_prob*100:.1f}%**")
#     # if pred_label == 1:
#     #      st.error("⚠️ Alto riesgo de diabetes")
#     # else:
#     #      st.success("✅ Bajo riesgo de diabetes")
#     # st.write(f"Cluster asignado: **{cluster_pred}**")
    


# # ==============================
# # PREDICCIÓN ACTUAL (SIEMPRE EXISTE)
# # ==============================

#     current_prediction = {
#         "Edad": age,
#         "IMC": bmi,
#         "Glucosa": gluc,
#         "Presion": bp,
#         "Outcome Predicho": "Alto riesgo" if pred_label == 1 else "Bajo riesgo",
#         "Probabilidad": round(pred_prob * 100, 1),
#         "Cluster": cluster_pred
#     }

#     # ==============================
#     # HISTORIAL SEGURO (SIN DUPLICADOS)
#     # ==============================
#     if "historial_predicciones" not in st.session_state:
#         st.session_state.historial_predicciones = pd.DataFrame(columns=current_prediction.keys())

#     if "last_pred_id" not in st.session_state:
#         st.session_state.last_pred_id = None

#     pred_id = f"{age}_{bmi}_{gluc}_{bp}_{pred_label}_{round(pred_prob,2)}"

#     if st.session_state.last_pred_id != pred_id:
#         st.session_state.historial_predicciones = pd.concat([
#             st.session_state.historial_predicciones,
#             pd.DataFrame([current_prediction])
#         ], ignore_index=True)

#         st.session_state.last_pred_id = pred_id

#     # ==============================
#     # MOSTRAR HISTORIAL
#     # ==============================
#     st.header("📊 Historial de predicciones")
#     st.dataframe(st.session_state.historial_predicciones)


# # ==============================
# # 📈 GRÁFICO CLÍNICO PROFESIONAL
# # ==============================

#     st.header("📈 Paciente vs Dataset (Vista Clínica)")

#     # 🎯 Selector
#     modo_vista = st.radio(
#         "Modo de visualización",
#         ["Actual", "Histórico"],
#         horizontal=True
#     )

#     # ==============================
#     # 📊 DATASET BASE (TENUE)
#     # ==============================
#     df_base = df_filtered.copy().reset_index(drop=True)
#     df_base["Cluster"] = df_base["Cluster"].astype(str)

#     fig = px.scatter(
#         df_base,
#         x="Glucose",
#         y="BMI",
#         color="Cluster",
#         opacity=0.35,  # 🔥 dataset en fondo
#         color_discrete_sequence=px.colors.qualitative.Set2
#     )

#     # ==============================
#     # 🧠 HISTORIAL (SEPARADO)
#     # ==============================
#     if "historial_grafico" not in st.session_state:
#         st.session_state.historial_grafico = pd.DataFrame(columns=["Glucose","BMI","Age","Cluster"])

#     if "last_graph_id" not in st.session_state:
#         st.session_state.last_graph_id = None

#     graph_id = f"{age}_{bmi}_{gluc}_{bp}_{pred_label}"

#     # guardar sin duplicar
#     if st.session_state.last_graph_id != graph_id:
#         new_patient = pd.DataFrame([{
#             "Glucose": gluc,
#             "BMI": bmi,
#             "Age": age,
#             "Cluster": str(cluster_pred)
#         }])

#         st.session_state.historial_grafico = pd.concat(
#             [st.session_state.historial_grafico, new_patient],
#             ignore_index=True
#         )

#         st.session_state.last_graph_id = graph_id

#     # mostrar histórico SOLO si se selecciona
#     if modo_vista == "Histórico" and not st.session_state.historial_grafico.empty:
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state.historial_grafico["Glucose"],
#                 y=st.session_state.historial_grafico["BMI"],
#                 mode="markers",
#                 marker=dict(
#                     size=9,
#                     color="yellow",
#                     opacity=0.9
#                 ),
#                 name="Pacientes ingresados"
#             )
#         )

#     # ==============================
#     # ⭐ PACIENTE ACTUAL
#     # ==============================
#     fig.add_trace(
#         go.Scatter(
#             x=[gluc],
#             y=[bmi],
#             mode="markers",
#             marker=dict(
#                 size=24,
#                 color="red",
#                 symbol="star",
#                 line=dict(width=2, color="white")
#             ),
#             name="Paciente actual"
#         )
#     )

#     # ==============================
#     # 🎨 LAYOUT PRO
#     # ==============================
#     fig.update_layout(
#         height=550,
#         template="plotly_dark",
#         title="Distribución clínica de pacientes",

#         legend=dict(
#             orientation="h",
#             y=1.02,
#             x=0.5,
#             xanchor="center"
#         ),

#         margin=dict(l=20, r=20, t=50, b=20),

#         xaxis=dict(
#             title="Glucosa (mg/dL)",
#             gridcolor="rgba(255,255,255,0.05)"
#         ),

#         yaxis=dict(
#             title="IMC (BMI)",
#             gridcolor="rgba(255,255,255,0.05)"
#         )
#     )

#     st.plotly_chart(fig, use_container_width=True)

# # ==============================
# # EXPORTACIÓN PDF + EXCEL
# # ==============================
#     import io
#     from fpdf import FPDF
#     import tempfile
#     import os

#     col1, col2 = st.columns(2)

#     # ---------------- PDF ----------------
#     with col1:
#         if st.button("📄 Generar PDF"):

#             pdf = FPDF()
#             pdf.add_page()
#             pdf.set_font("Arial", "B", 16)
#             pdf.cell(0, 10, "Reporte de Diabetes", ln=True, align="C")
#             pdf.ln(10)

#             pdf.set_font("Arial", "", 12)

#             for k, v in current_prediction.items():
#                 pdf.cell(0, 8, f"{k}: {v}", ln=True)

#             pdf.ln(5)

#             # gráfico
#             # plt.figure(figsize=(6,4))
#             # for c in sorted(df_plot["Cluster"].unique()):
#             #     sub = df_plot[df_plot["Cluster"] == c]
#             #     plt.scatter(sub["Glucose"], sub["BMI"], label=f"Cluster {c}")

#             # plt.xlabel("Glucosa")
#             # plt.ylabel("BMI")
#             # plt.legend()
#             plt.figure(figsize=(6,4))

#             # Dataset base
#             for c in sorted(df_base["Cluster"].unique()):
#                 sub = df_base[df_base["Cluster"] == c]
#                 plt.scatter(
#                     sub["Glucose"],
#                     sub["BMI"],
#                     alpha=0.3,
#                     label=f"Cluster {c}"
#                 )

#             # Histórico (si existe)
#             if "historial_grafico" in st.session_state and not st.session_state.historial_grafico.empty:
#                 plt.scatter(
#                     st.session_state.historial_grafico["Glucose"],
#                     st.session_state.historial_grafico["BMI"],
#                     color="yellow",
#                     label="Histórico"
#                 )

#             # Paciente actual
#             plt.scatter(
#                 gluc,
#                 bmi,
#                 color="red",
#                 s=120,
#                 label="Paciente actual"
#             )

#             plt.xlabel("Glucosa")
#             plt.ylabel("BMI")
#             plt.title("Paciente vs Dataset")
#             plt.legend()

#             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#                 plt.savefig(tmp.name)
#                 tmp_path = tmp.name

#             plt.close()

#             pdf.image(tmp_path, x=10, w=180)
#             os.remove(tmp_path)

#             pdf_bytes = pdf.output(dest="S").encode("latin1")

#             st.download_button(
#                 "⬇️ Descargar PDF",
#                 data=pdf_bytes,
#                 file_name="reporte_diabetes.pdf",
#                 mime="application/pdf"
#             )

#     # ---------------- EXCEL ----------------
#     with col2:
#         if st.button("📊 Exportar Excel"):

#             output = io.BytesIO()

#             with pd.ExcelWriter(output, engine="openpyxl") as writer:

#                 # 1. pacientes filtrados
#                 df_filtered.to_excel(writer, sheet_name="Pacientes", index=False)

#                 # 2. paciente actual
#                 pd.DataFrame([current_prediction]).to_excel(
#                     writer,
#                     sheet_name="Nuevo Paciente",
#                     index=False
#                 )

#                 # 3. historial
#                 st.session_state.historial_predicciones.to_excel(
#                     writer,
#                     sheet_name="Historial",
#                     index=False
#                 )

#                 # 4. resumen clusters
#                 cluster_summary.to_excel(writer, sheet_name="Clusters")

#             output.seek(0)

#             st.download_button(
#                 "⬇️ Descargar Excel",
#                 data=output,
#                 file_name="dashboard_diabetes.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
              

# ==============================
# app.py - Dashboard Diabetes Interactivo (Refactor PRO)
# ==============================




# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import io
# from fpdf import FPDF
# import tempfile
# import os

# # 🔥 NUEVA ARQUITECTURA
# from ml.training import load_and_train
# from ml.prediction import predict_patient
# from ml.clustering import build_clusters
# from state.history import update_history

# # ==============================
# # CONFIG
# # ==============================



# st.set_page_config(page_title="Dashboard Diabetes Interactivo 🏥", layout="wide")

# # ==============================
# # HEADER
# # ==============================
# st.markdown("""
# <h1 style='text-align: center; font-size:25px; color: #00BFA6;'>
# 🏥 Predicción Inteligente de Diabetes
# </h1>
# <p style='text-align: center; font-size:18px;'>
# Predicción • Segmentación • Análisis en tiempo real
# </p>
# """, unsafe_allow_html=True)

# # ==============================
# # ESTILOS
# # ==============================
# st.markdown("""
# <style>
# h2 {
#     font-size: 18px !important;
#     font-weight: 600;
#     color: #E5E7EB;
#     padding-left: 10px;
# }
# .kpi {
#     background-color: #1f2937;
#     padding: 10px;
#     border-radius: 8px;
#     text-align: center;
#     color: white;
#     border-left: 4px solid #00BFA6;
# }
# .kpi h2 { font-size: 22px; margin: 0; }
# .kpi h3 { font-size: 14px; margin: 0; color: #9CA3AF; }
# </style>
# """, unsafe_allow_html=True)

# # ==============================
# # 🔥 CARGA + ENTRENAMIENTO (MODULAR)
# # ==============================
# df_orig, X_scaled, y, scaler, rf_model, results = load_and_train()

# # ==============================
# # 🔥 CLUSTERING
# # ==============================
# kmeans, pca, X_pca, df_orig = build_clusters(X_scaled, df_orig)

# cluster_summary = df_orig.groupby('Cluster').mean()

# # ==============================
# # SIDEBAR FILTROS
# # ==============================
# st.sidebar.header("Filtros de pacientes")

# df_dynamic = df_orig.copy()

# age_filter = st.sidebar.slider("Edad", int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max()), 
#                                (int(df_dynamic['Age'].min()), int(df_dynamic['Age'].max())))

# bmi_filter = st.sidebar.slider("IMC (BMI)", float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max()), 
#                                (float(df_dynamic['BMI'].min()), float(df_dynamic['BMI'].max())))

# glucose_filter = st.sidebar.slider("Glucosa", int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max()), 
#                                    (int(df_dynamic['Glucose'].min()), int(df_dynamic['Glucose'].max())))

# cluster_filter = st.sidebar.multiselect("Cluster", options=sorted(df_dynamic['Cluster'].unique()), 
#                                         default=sorted(df_dynamic['Cluster'].unique()))

# df_filtered = df_dynamic[
#     (df_dynamic['Age'] >= age_filter[0]) & (df_dynamic['Age'] <= age_filter[1]) &
#     (df_dynamic['BMI'] >= bmi_filter[0]) & (df_dynamic['BMI'] <= bmi_filter[1]) &
#     (df_dynamic['Glucose'] >= glucose_filter[0]) & (df_dynamic['Glucose'] <= glucose_filter[1]) &
#     (df_dynamic['Cluster'].isin(cluster_filter))
# ]

# # ==============================
# # KPIs
# # ==============================
# st.header("📊 Indicadores Clave")

# col1, col2, col3, col4 = st.columns(4)

# col1.markdown(f"<div class='kpi'><h3>Pacientes</h3><h2>{len(df_filtered)}</h2></div>", unsafe_allow_html=True)
# col2.markdown(f"<div class='kpi'><h3>Riesgo Promedio</h3><h2>{df_filtered['Outcome'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
# col3.markdown(f"<div class='kpi'><h3>Glucosa Promedio</h3><h2>{df_filtered['Glucose'].mean():.1f}</h2></div>", unsafe_allow_html=True)
# col4.markdown(f"<div class='kpi'><h3>Clusters</h3><h2>{df_filtered['Cluster'].nunique()}</h2></div>", unsafe_allow_html=True)

# # ==============================
# # INSIGHT
# # ==============================
# st.header("🧠 Insight automático")

# riesgo = df_filtered['Outcome'].mean()

# if riesgo > 0.6:
#     st.error("⚠️ Alto riesgo general en la población analizada")
# elif riesgo > 0.3:
#     st.warning("⚠️ Riesgo moderado detectado")
# else:
#     st.success("✅ Bajo riesgo general")

# # ==============================
# # TABS
# # ==============================
# tabs = st.tabs(["Datos & EDA", "Distribución", "Correlación",
#                 "Modelos", "Random Forest", "Clustering", "Predicción"])

# # ---------------- TAB 1 ----------------
# with tabs[0]:
#     st.dataframe(df_filtered)
#     st.write(df_filtered.describe())

# # ---------------- TAB 2 ----------------
# with tabs[1]:
#     fig = px.histogram(df_filtered, x='Outcome', color='Outcome')
#     st.plotly_chart(fig, use_container_width=True)

# # ---------------- TAB 3 ----------------
# with tabs[2]:
#     corr = df_filtered.corr()
#     fig = px.imshow(corr, text_auto=True)
#     st.plotly_chart(fig, use_container_width=True)

# # ---------------- TAB 4 ----------------
# with tabs[3]:
#     results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Precision"])
#     st.dataframe(results_df)

# # ---------------- TAB 5 ----------------
# with tabs[4]:
#     importances = pd.Series(rf_model.feature_importances_, index=df_orig.drop("Outcome", axis=1).columns)
#     fig = px.bar(importances)
#     st.plotly_chart(fig)

# # ---------------- TAB 6 ----------------
# with tabs[5]:
#     fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=df_orig['Cluster'].astype(str))
#     st.plotly_chart(fig)

# # ---------------- TAB 7 ----------------
# with tabs[6]:

#     preg = st.sidebar.slider("Embarazos", 0, 20, 1)
#     gluc = st.sidebar.slider("Glucosa", 0, 200, 120)
#     bp = st.sidebar.slider("Presión", 0, 140, 70)
#     skin = st.sidebar.slider("Piel", 0, 100, 20)
#     ins = st.sidebar.slider("Insulina", 0, 900, 80)
#     bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
#     dpf = st.sidebar.slider("DPF", 0.0, 3.0, 0.5)
#     age = st.sidebar.slider("Edad", 10, 100, 40)

#     pred_label, pred_prob = predict_patient(
#         [preg, gluc, bp, skin, ins, bmi, dpf, age],
#         scaler,
#         rf_model
#     )

#     cluster_pred = kmeans.predict(scaler.transform([[preg, gluc, bp, skin, ins, bmi, dpf, age]]))[0]

#     st.metric("Probabilidad", f"{pred_prob*100:.1f}%")
#     st.metric("Cluster", cluster_pred)
#     st.metric("Riesgo", "Alto" if pred_label else "Bajo")

#     # HISTORIAL
#     prediction = {
#         "Edad": age,
#         "IMC": bmi,
#         "Glucosa": gluc,
#         "Outcome": "Alto" if pred_label else "Bajo",
#         "Probabilidad": round(pred_prob * 100, 1)
#     }

#     pred_id = f"{age}_{bmi}_{gluc}_{pred_label}"

#     update_history(st.session_state, prediction, pred_id)

#     st.dataframe(st.session_state.historial_predicciones)              



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Dashboard Diabetes Interactivo 🏥",
    layout="wide"
)

# ==============================
# CARGA MODELO (CACHE PARA PERFORMANCE)
# ==============================
@st.cache_resource
def load_assets():
    model = joblib.load("ml/artifacts/model.pkl")
    scaler = joblib.load("ml/artifacts/scaler.pkl")
    return model, scaler

rf_model, scaler = load_assets()

# ==============================
# DATASET (PUEDES CAMBIAR A API EN FUTURO)
# ==============================
df_orig = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)

# limpieza simple
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_orig[cols] = df_orig[cols].replace(0, np.nan)
df_orig[cols] = df_orig[cols].fillna(df_orig[cols].median())

X = df_orig.drop("Outcome", axis=1)
y = df_orig["Outcome"]

# ==============================
# CLUSTERING (CACHE)
# ==============================
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@st.cache_resource
def build_clusters(X_scaled, df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["Cluster"] = clusters

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return kmeans, pca, X_pca, df

# ==============================
# ESCALADO
# ==============================
from sklearn.preprocessing import StandardScaler

scaler_local = StandardScaler()
X_scaled = scaler_local.fit_transform(X)

kmeans, pca, X_pca, df_orig = build_clusters(X_scaled, df_orig)

# ==============================
# UI HEADER
# ==============================
st.markdown("""
<h1 style='text-align: center; color: #00BFA6;'>
🏥 Predicción Inteligente de Diabetes
</h1>
<p style='text-align: center; font-size:18px;'>
ML • Clustering • Analítica Clínica
</p>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR FILTROS
# ==============================
st.sidebar.header("Filtros")

age_filter = st.sidebar.slider(
    "Edad",
    int(df_orig["Age"].min()),
    int(df_orig["Age"].max()),
    (20, 60)
)

bmi_filter = st.sidebar.slider(
    "BMI",
    float(df_orig["BMI"].min()),
    float(df_orig["BMI"].max()),
    (20.0, 40.0)
)

glucose_filter = st.sidebar.slider(
    "Glucosa",
    int(df_orig["Glucose"].min()),
    int(df_orig["Glucose"].max()),
    (80, 180)
)

cluster_filter = st.sidebar.multiselect(
    "Cluster",
    options=sorted(df_orig["Cluster"].unique()),
    default=sorted(df_orig["Cluster"].unique())
)

df_filtered = df_orig[
    (df_orig["Age"].between(age_filter[0], age_filter[1])) &
    (df_orig["BMI"].between(bmi_filter[0], bmi_filter[1])) &
    (df_orig["Glucose"].between(glucose_filter[0], glucose_filter[1])) &
    (df_orig["Cluster"].isin(cluster_filter))
]

# ==============================
# KPIs
# ==============================
st.header("📊 KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Pacientes", len(df_filtered))
col2.metric("Riesgo Promedio", f"{df_filtered['Outcome'].mean()*100:.1f}%")
col3.metric("Glucosa Prom", f"{df_filtered['Glucose'].mean():.1f}")
col4.metric("Clusters", df_filtered["Cluster"].nunique())

# ==============================
# INSIGHT
# ==============================
st.header("🧠 Insight")

risk = df_filtered["Outcome"].mean()

if risk > 0.6:
    st.error("⚠️ Alto riesgo poblacional")
elif risk > 0.3:
    st.warning("⚠️ Riesgo moderado")
else:
    st.success("✅ Bajo riesgo")

# ==============================
# TABS
# ==============================
tabs = st.tabs([
    "Datos",
    "Distribución",
    "Correlación",
    "Modelos",
    "Importancia",
    "Clustering",
    "Predicción"
])

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.dataframe(df_filtered)

# ---------------- TAB 2 ----------------
with tabs[1]:
    fig = px.histogram(df_filtered, x="Outcome", color="Outcome")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3 ----------------
with tabs[2]:
    fig = px.imshow(df_filtered.corr(), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.write("Modelo cargado desde backend (Random Forest listo)")

# ---------------- TAB 5 ----------------
with tabs[4]:
    importances = pd.Series(
        rf_model.feature_importances_,
        index=df_orig.drop("Outcome", axis=1).columns
    )

    fig = px.bar(importances.sort_values())
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 6 ----------------
with tabs[5]:
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=df_orig["Cluster"].astype(str)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 7 ----------------
with tabs[6]:

    st.subheader("🧪 Nuevo paciente")

    preg = st.slider("Embarazos", 0, 20, 1)
    gluc = st.slider("Glucosa", 0, 200, 120)
    bp = st.slider("Presión", 0, 140, 70)
    skin = st.slider("Piel", 0, 100, 20)
    ins = st.slider("Insulina", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("DPF", 0.0, 3.0, 0.5)
    age = st.slider("Edad", 10, 100, 40)

    # ==============================
    # PREDICCIÓN
    # ==============================
    X_new = scaler.transform([[preg, gluc, bp, skin, ins, bmi, dpf, age]])

    prob = rf_model.predict_proba(X_new)[0][1]
    pred = int(prob > 0.5)

    cluster = kmeans.predict(X_new)[0]

    # ==============================
    # RESULTADOS
    # ==============================
    st.metric("Probabilidad", f"{prob*100:.1f}%")
    st.metric("Cluster", cluster)
    st.metric("Riesgo", "Alto" if pred else "Bajo")

    if pred:
        st.error("⚠️ Alto riesgo de diabetes")
    else:
        st.success("✅ Bajo riesgo de diabetes")

    # ==============================
    # HISTORIAL LOCAL (TEMPORAL)
    # ==============================
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Edad": age,
        "Glucosa": gluc,
        "BMI": bmi,
        "Probabilidad": round(prob * 100, 1),
        "Riesgo": "Alto" if pred else "Bajo",
        "Cluster": cluster
    })

    st.subheader("📊 Historial de predicciones")
    st.dataframe(pd.DataFrame(st.session_state.history))