@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install uv and sync this project before running it.
    exit /b 1
)

echo.
echo ^>^>^> uv run python -m adaptive_rag.data
uv run python -m adaptive_rag.data
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m adaptive_rag.indexes
uv run python -m adaptive_rag.indexes
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m pytest tests -v
uv run python -m pytest tests -v
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m evaluation.run_eval --split dev --mode all --tag three_way
uv run python -m evaluation.run_eval --split dev --mode all --tag three_way
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m evaluation.precompute_kcurve --split dev
uv run python -m evaluation.precompute_kcurve --split dev
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m evaluation.precompute_kcurve --split test
uv run python -m evaluation.precompute_kcurve --split test
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv run python -m evaluation.timing --split test --repeats 3
uv run python -m evaluation.timing --split test --repeats 3
if errorlevel 1 goto :failed

echo.
echo ^>^>^> uv export --no-hashes -o requirements.txt
uv export --no-hashes -o requirements.txt
if errorlevel 1 goto :failed

echo.
echo All pipeline commands completed successfully.
exit /b 0

:failed
echo.
echo Pipeline stopped because a command failed.
exit /b 1
