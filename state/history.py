import pandas as pd

def update_history(session_state, prediction, pred_id):
    if "historial_predicciones" not in session_state:
        session_state.historial_predicciones = pd.DataFrame(columns=prediction.keys())

    if "last_pred_id" not in session_state:
        session_state.last_pred_id = None

    if session_state.last_pred_id != pred_id:
        session_state.historial_predicciones = pd.concat(
            [session_state.historial_predicciones, pd.DataFrame([prediction])],
            ignore_index=True
        )
        session_state.last_pred_id = pred_id