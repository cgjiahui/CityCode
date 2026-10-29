param(
    [Parameter(Mandatory=$true)][string]$JobFile,
    [string]$EditorExe = 'D:\files\ue_assets\UE_5.0\Engine\Binaries\Win64\UnrealEditor-Cmd.exe',
    [string]$Project = 'D:\files\ue_assets\CitySample\CitySample.uproject',
    [string]$LocalDataCachePath,
    [switch]$SkipShaderCompilation
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$jobPath = (Resolve-Path -LiteralPath $JobFile).Path
$job = Get-Content -Raw -Encoding UTF8 -LiteralPath $jobPath | ConvertFrom-Json
# quit_after_execution declares task ownership for this controller. The engine's
# ExecutePythonScript mode owns shutdown; the Python entry never requests quit.
if (!$job.headless -or !$job.quit_after_execution) { throw 'This controller requires an explicitly owned, task-completion-exiting editor job.' }
if (!(Test-Path -LiteralPath $EditorExe) -or !(Test-Path -LiteralPath $Project)) { throw 'Existing engine/project not found.' }
if (Test-Path -LiteralPath $job.output_json) { throw 'Use a fresh result path; previous results are retained.' }
$existing = @(Get-Process -ErrorAction Stop | Where-Object { $_.ProcessName -match '^UnrealEditor(-Cmd)?$|^ShaderCompileWorker$' })
if ($existing.Count) { throw 'An Unreal editor or shader worker is running. Leave the current session untouched and run this job after it exits.' }
$script = Join-Path $PSScriptRoot 'apply_elevated_repair.py'
if (!(Test-Path -LiteralPath $script)) { throw 'Repair entry is missing beside this controller.' }
$env:UE_CITYCODE_REPAIR_JOB = $jobPath
if (![string]::IsNullOrWhiteSpace($LocalDataCachePath)) {
    [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $LocalDataCachePath, [EnvironmentVariableTarget]::Process)
}
$work = Split-Path -Parent $jobPath
$log = Join-Path $work 'apply_editor.log'
$arguments = @($Project, '/Engine/Maps/Entry', '-unattended', '-nosplash', '-nosound', '-nop4', '-nosourcecontrol', '-NullRHI',
    '-EnablePlugins=GLTFImporter,EditorScriptingUtilities,PythonScriptPlugin', '-corelimit=8',
    '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False',
    ('-ExecutePythonScript=' + $script), ('-abslog=' + $log))
if ($SkipShaderCompilation) { $arguments += '-NoShaderCompile' }
# No timer terminates this process. ExecutePythonScript exits after Python returns;
# an explicit Python quit would request a second teardown. Exit-code checks stay strict.
& $EditorExe @arguments
$editorExit = $LASTEXITCODE
if (!(Test-Path -LiteralPath $job.output_json)) { throw "Editor exited without a repair result: $editorExit. Inspect $log" }
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $job.output_json | ConvertFrom-Json
if ($editorExit -ne 0 -or $result.status -eq 'failed') { throw "Repair failed; baseline is retained. Inspect $($job.output_json)" }
[pscustomobject]@{ status=$result.status; new_level_path=$result.new_level_path; actor_count=$result.actor_count; independent_readback_required=$true } | ConvertTo-Json -Compress
