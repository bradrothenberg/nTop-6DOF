@echo off
REM Run AVL analysis for conventional tail configuration
REM This batch file automates AVL execution

cd /d "C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF\avl_files"

echo Running AVL for conventional tail configuration...
echo.

"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe" < avl_commands.txt

echo.
echo Done! Check for output files:
echo   - conventional_trim.ft
echo   - conventional_trim.st
echo.
pause
