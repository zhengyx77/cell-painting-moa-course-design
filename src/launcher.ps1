param([ValidateSet('demo','reproduce','verify')][string]$Mode='demo')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$localPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
$bundledPython=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if(Test-Path -LiteralPath $localPython){$pythonExe=$localPython}
elseif(Test-Path -LiteralPath $bundledPython){$pythonExe=$bundledPython}
else {$pythonExe=(Get-Command python -ErrorAction Stop).Source}
function Run-ProjectScript([string]$scriptName){
    & $pythonExe (Join-Path $PSScriptRoot $scriptName)
    if($LASTEXITCODE -ne 0){throw "Failed: $scriptName. See README.md for environment setup."}
}
switch($Mode){
    'demo' {Run-ProjectScript 'serve_demo.py'}
    'verify' {Run-ProjectScript 'verify_project.py'}
    'reproduce' {
        Run-ProjectScript 'download_data.py'
        Run-ProjectScript 'run_experiment.py'
        Run-ProjectScript 'analyze_results.py'
        Run-ProjectScript 'build_demo.py'
        Run-ProjectScript 'verify_project.py'
    }
}
