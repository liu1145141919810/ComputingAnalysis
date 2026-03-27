@echo off
setlocal enabledelayedexpansion


:: ===================== Clean Solution-Level Temp Files =====================
echo [Cleaning solution-level files...]

:: Delete .vs hidden folder (solution config)
if exist ".vs\" (
    echo   Deleting .vs folder...
    rmdir /s /q ".vs"
)

:: Delete solution-level build output dirs (e.g., x64/Debug)
set "solution_build_dirs=x64 Debug Release"
for %%d in (%solution_build_dirs%) do (
    if exist "%%d\" (
        echo   Deleting solution-level %%d folder...
        rmdir /s /q "%%d"
    )
)

:: Delete solution-level temp files
del /q /f /s *.suo 2>nul       :: Solution user options
del /q /f /s *.VC.db 2>nul     :: VS database files
del /q /f /s *.VC.opendb 2>nul :: VS database files

:: ===================== Clean Each Project Folder =====================
echo [Cleaning files in each project folder...]

:: Loop through all project subdirectories in the solution root
for /d %%p in (*) do (
    echo.
    echo   Cleaning project: %%p
    
    :: Delete project's obj folder (compiled objects)
    if exist "%%p\obj\" (
        echo     Deleting project's obj directory...
        rmdir /s /q "%%p\obj"
    )
    
    :: Delete project's build output dirs (Debug/Release)
    set "project_build_dirs=Debug Release"
    for %%c in (%project_build_dirs%) do (
        if exist "%%p\%%c\" (
            echo     Deleting project's %%c directory...
            rmdir /s /q "%%p\%%c"
        )
    )
    
    :: Delete project's platform-specific output (x64/Debug, x64/Release)
    if exist "%%p\x64\" (
        for %%c in (Debug Release) do (
            if exist "%%p\x64\%%c\" (
                echo     Deleting project's x64\%%c directory...
                rmdir /s /q "%%p\x64\%%c"
            )
        )
    )
    
    :: Delete intermediate files in project root
    del /q /f /s "%%p\*.obj" 2>nul  :: Compiled object files
    del /q /f /s "%%p\*.pch" 2>nul  :: Precompiled headers
    del /q /f /s "%%p\*.ilk" 2>nul  :: Linker incremental files
    del /q /f /s "%%p\*.res" 2>nul  :: Uncompiled resource files
    del /q /f /s "%%p\*.sbr" 2>nul  :: Browse info files
    del /q /f /s "%%p\*.idb" 2>nul  :: Intermediate databases
    
    :: Delete project user options
    del /q /f /s "%%p\*.user" 2>nul :: Project-specific configs
)

:: ===================== Completion Message =====================
echo.
echo All temporary files in the solution and projects have been cleaned up!
pause