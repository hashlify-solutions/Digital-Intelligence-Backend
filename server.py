from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from setup import setup_logging, setup_application
from routers import user, alert, case, test, processing_profile, platform_control, detector
from routers.v1 import case_v1, ufdr_metadata_v1, media_files_v1,neo4j_sync_v1, llama_model_v1
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from config.settings import settings
import warnings

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to download models before server starts"""
    await setup_application()
    yield  # Continue running the app

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # List of allowed origins from settings
    allow_credentials=True,  # Allow cookies or authorization headers
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
logger.info(f"Models will be downloaded locally: {settings.local_models}")

# Create required directories
Path(settings.upload_dir).mkdir(exist_ok=True)  # Create the directory if it doesn't exist
Path(settings.logo_dir).mkdir(exist_ok=True)  # Create the logo directory if it doesn't exist

# Mount the logo and data directories to serve static files
app.mount("/logo", StaticFiles(directory=settings.logo_dir), name="logo")
app.mount("/data", StaticFiles(directory=settings.upload_dir), name="data")

# add routes
app.include_router(user.router, prefix="/users", tags=["Users"])
app.include_router(alert.alert_router, prefix="/alerts", tags=["Alerts"])
app.include_router(case.router, prefix="/case", tags=["Case"])
app.include_router(processing_profile.processing_router, prefix="/models-profile", tags=["Models Profile"])
app.include_router(test.router, prefix="/test", tags=["Test"])
app.include_router(platform_control.router, prefix="/platform-control", tags=["Platform Control"])
app.include_router(detector.router, prefix="/detectors", tags=["Detectors"])
# add v1 routes
app.include_router(case_v1.router, prefix="/v1/case", tags=["Case"])
app.include_router(llama_model_v1.router, prefix="/v1/llama", tags=["Llama Model"])
app.include_router(ufdr_metadata_v1.router, prefix="/v1/ufdr", tags=["UFDR Metadata"])
app.include_router(media_files_v1.router, prefix="/v1/api/media", tags=["Media Files"])
app.include_router(neo4j_sync_v1.router, prefix="/v1/neo4j", tags=["Neo4j"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        log_level=settings.log_level.lower()
    )