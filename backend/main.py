# from fastapi import FastAPI
# from backend.routes import auth_routes, predict_routes, user_routes

# app = FastAPI(title="Diabetes ML API")

# # Registrar rutas
# app.include_router(auth_routes.router)
# app.include_router(predict_routes.router)
# app.include_router(user_routes.router)

# @app.get("/")
# def root():
#     return {"message": "Diabetes ML API running"}


#PARA RENDER
from fastapi import FastAPI
from backend.routes import auth_routes, predict_routes, user_routes

app = FastAPI(title="Diabetes ML API")

# Registrar rutas
app.include_router(auth_routes.router)
app.include_router(predict_routes.router)
app.include_router(user_routes.router)


@app.get("/")
def root():
    return {"message": "API Diabetes ML funcionando 🚀"}