@echo off
setlocal EnableExtensions
set "LBM_EMBEDDED_REPO=%~dp0"
set "LBM_EMBEDDED_PROJECT_ID=las_cafiisica_ground_v2"
call "%~dp0tools\LocalBuildManager_CLASSIFY_LAS\START_LOCAL_BUILD_MANAGER.bat"
exit /b %ERRORLEVEL%
