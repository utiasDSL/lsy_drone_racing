@echo off
setlocal EnableExtensions

if "%PIXI_PROJECT_ROOT%"=="" (
  call :log Not running inside a Pixi environment; skipping setup_acados.bat
  exit /b 0
)

if not exist "%PIXI_PROJECT_ROOT%\pixi.lock" (
  call :log ERROR: pixi environment is not properly set up.
  exit /b 0
)

set "ACADOS_DIR=%PIXI_PROJECT_ROOT%\acados"
set "ACADOS_DLL="

if exist "%ACADOS_DIR%\bin\acados.dll" set "ACADOS_DLL=%ACADOS_DIR%\bin\acados.dll"
if exist "%ACADOS_DIR%\bin\libacados.dll" set "ACADOS_DLL=%ACADOS_DIR%\bin\libacados.dll"
if exist "%ACADOS_DIR%\lib\acados.dll" set "ACADOS_DLL=%ACADOS_DIR%\lib\acados.dll"
if exist "%ACADOS_DIR%\lib\libacados.dll" set "ACADOS_DLL=%ACADOS_DIR%\lib\libacados.dll"

if defined ACADOS_DLL (
  set "ACADOS_SOURCE_DIR=%ACADOS_DIR%"
  set "ACADOS_INSTALL_DIR=%ACADOS_DIR%"
  set "PATH=%ACADOS_DIR%\bin;%ACADOS_DIR%\interfaces\acados_template;%PATH%"
  call :log Using existing native Windows acados installation at "%ACADOS_DIR%".
  exit /b 0
)

call :log Native Windows acados bootstrap is not supported by this repository.
call :log Pixi shell is ready without acados. Use WSL if you need MPC or acados-based controllers.
exit /b 0

:log
echo [Setup Acados] %*
exit /b 0
