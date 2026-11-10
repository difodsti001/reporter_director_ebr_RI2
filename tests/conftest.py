import os
import sys

# Permite `import core` sin importar desde qué directorio se invoque pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# core.py y llm_client.py leen estas variables al importarse (para armar
# los pools de conexión y el cliente de Azure OpenAI). En los tests solo
# se ejercitan funciones puras (nunca se abren los pools ni se llama al
# LLM), así que basta con valores dummy para que el import no falle.
os.environ.setdefault("DSN_DATOS", "postgresql://user:pass@localhost/db_test")
os.environ.setdefault("DSN_CACHE", "postgresql://user:pass@localhost/db_test")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
