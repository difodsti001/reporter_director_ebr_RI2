"""
run.py

Punto de arranque para Windows. Reemplaza al comando `uvicorn app:app`.
Uso:
    python run.py
"""

import sys
import asyncio

import uvicorn


async def main():
    config = uvicorn.Config("app:app", host="127.0.0.1", port=8000, reload=False)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())