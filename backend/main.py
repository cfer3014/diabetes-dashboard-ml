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
import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)