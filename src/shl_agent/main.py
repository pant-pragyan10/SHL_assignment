from shl_agent.api import app

# Ensure routes defined in `shl_agent.api.app` (APIRouter `router`) are
# included on the FastAPI `app` used by uvicorn. This avoids import-order
# issues where the package import didn't register submodule routes.
try:
	from shl_agent.api import routes as api_routes_module
	if hasattr(api_routes_module, "router"):
		app.include_router(api_routes_module.router)
except Exception:
	pass

# Expose `app` for uvicorn: `uvicorn shl_agent.main:app`
