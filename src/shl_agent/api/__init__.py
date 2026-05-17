"""API package for the SHL recommendation agent.

Creates the FastAPI `app` and imports the `routes` module so handlers are
registered when the package is imported (necessary for uvicorn entrypoints).
"""
from fastapi import FastAPI, Request
import time
import logging
import json
import os
import inspect
from shl_agent.logging_config import configure_logging


app = FastAPI(title="SHL Recommendation Agent")

# Configure logging early
configure_logging()

# timing middleware: lightweight instrumentation for all requests
_log = logging.getLogger(__name__)


@app.middleware('http')
async def _timing_middleware(request: Request, call_next):
	start = time.time()
	resp = await call_next(request)
	elapsed = time.time() - start
	resp.headers['X-Process-Time'] = f"{elapsed:.3f}s"
	if elapsed > 2.0:
		_log.info('Slow request %s %s elapsed=%.3fs', request.method, request.url.path, elapsed)
	return resp


# Import module that defines an APIRouter and include it so handlers are
# registered on the shared FastAPI `app` instance used by uvicorn.
from . import routes as _routes  # import routes module exposing `router`
try:
	app.include_router(_routes.router)
except Exception:
	_log.exception('Failed to include routes')


def _print_startup_info() -> None:
	"""Log basic startup information and registered routes."""
	mod = __name__
	try:
		_log.info('=== SHL AGENT STARTUP ===')
		_log.info('app.title: %s', getattr(app, 'title', None))
		_log.info('module file: %s', inspect.getfile(inspect.getmodule(object)))
	except Exception:
		_log.exception('Error during startup info logging')

	# list registered routes at debug level
	for r in app.routes:
		try:
			_log.debug('route: %s %s', r.path, getattr(r, 'methods', None))
		except Exception:
			_log.debug('route: %s', getattr(r, 'path', 'unknown'))


@app.on_event('startup')
def _startup_event():
	_print_startup_info()

	# Initialize shared Groq client if API key present; store on app.state for reuse.
	try:
		from shl_agent.llm.groq_client import GroqClient
	except Exception:
		GroqClient = None

	try:
		# ensure .env variables are loaded into environment when present
		from dotenv import load_dotenv
		load_dotenv()
	except Exception:
		pass

	api_key = os.getenv('GROQ_API_KEY')
	try:
		if api_key:
			_log.info('GROQ_API_KEY found in environment')
		else:
			_log.info('GROQ_API_KEY not found in environment')
	except Exception:
		_log.exception('Error checking GROQ_API_KEY presence')

	if api_key and GroqClient is not None:
		try:
			app.state.groq_client = GroqClient(api_key)
			_log.info('GroqClient initialized and stored on app.state.groq_client')
		except Exception:
			app.state.groq_client = None
			_log.exception('Failed to initialize GroqClient at startup')
	else:
		app.state.groq_client = None
		_log.warning('GROQ API client unavailable; Groq features disabled')

	# Validate catalog presence and basic sanity
	catalog_path = os.path.join(os.getcwd(), 'data', 'processed', 'catalog.json')
	if os.path.exists(catalog_path):
		try:
			with open(catalog_path, 'r', encoding='utf-8') as f:
				cat = json.load(f)
			if not isinstance(cat, list) or len(cat) == 0:
				_log.warning('Catalog exists but is empty or malformed')
		except Exception:
			_log.exception('Failed to parse catalog.json during startup')
	else:
		_log.warning('Catalog not present at %s; /chat will return empty recommendations', catalog_path)

	# Optionally preload embeddings to reduce cold-start latency if requested
	if os.getenv('PRELOAD_EMBEDDINGS', 'false').lower() in ('1', 'true', 'yes'):
		try:
			from shl_agent.retrieval.retriever import HybridRetriever
			with open(catalog_path, 'r', encoding='utf-8') as f:
				cat = json.load(f)
			app.state.retriever = HybridRetriever(cat)
			_log.info('Preloaded retriever and embeddings into app.state.retriever')
		except Exception:
			_log.exception('Failed to preload embeddings at startup')