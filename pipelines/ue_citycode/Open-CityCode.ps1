param(
    [Parameter(Mandatory=$true)][string]$EditorExe,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][string]$AssemblyResult,
    [Parameter(Mandatory=$true)][string]$ReadbackResult,
    [string]$LocalDataCachePath
)
$ErrorActionPreference = 'Stop'
function Quote-WindowsArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    # Match Start-CityCode: Start-Process joins ArgumentList into one Windows
    # command line, so embedded Python-path quotes need argv escaping too.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}
if ($env:SSH_CONNECTION) { throw 'Run Open-CityCode.cmd from the signed-in Windows desktop. An SSH desktop process is not visible in your interactive session.' }
if (!(Test-Path -LiteralPath $EditorExe)) { throw "Editor not found: $EditorExe" }
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $AssemblyResult | ConvertFrom-Json
$readback = Get-Content -Raw -Encoding UTF8 -LiteralPath $ReadbackResult | ConvertFrom-Json
$assemblyHash = (Get-FileHash -LiteralPath $AssemblyResult -Algorithm SHA256).Hash.ToLowerInvariant()
if ($readback.assembly_result_sha256 -ne $assemblyHash -or $readback.assembly_manifest_sha256 -ne $result.assembly_manifest_sha256) {
    throw 'Assembly/readback files are stale or do not describe the same frozen scene.'
}
if (!$result.ue_executed -or !$readback.ue_executed -or !$readback.independent_saved_map_readback -or
    $readback.status -ne 'passed' -or @($readback.errors).Count -ne 0 -or
    !$readback.inspection_lighting_checked -or !$readback.generated_normals_checked -or
    $readback.new_level_path -ne $result.new_level_path) {
    throw 'This scene needs matching independent saved-map verification with no errors and checked inspection lighting and generated normals.'
}
if (!$result.new_level_path.StartsWith('/Game/Research/CityCodeOpenSource/Maps/')) { throw 'Unexpected map namespace.' }
$resolvedDataCachePath = $LocalDataCachePath
if ([string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    foreach ($environmentScope in @([EnvironmentVariableTarget]::Process, [EnvironmentVariableTarget]::User, [EnvironmentVariableTarget]::Machine)) {
        $candidateDataCachePath = [Environment]::GetEnvironmentVariable('UE-LocalDataCachePath', $environmentScope)
        if (![string]::IsNullOrWhiteSpace($candidateDataCachePath)) { $resolvedDataCachePath = $candidateDataCachePath; break }
    }
}
if ([string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    $projectLogRoot = Join-Path ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Project))) 'Saved\Logs'
    foreach ($guiLog in @(Get-ChildItem -LiteralPath $projectLogRoot -Filter '*.log' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)) {
        try {
            $cacheEvidence = @(Select-String -LiteralPath $guiLog.FullName -Pattern 'Command Line:', 'Using Local data cache path (.+?): Writable' -ErrorAction Stop)
            if (@($cacheEvidence | Where-Object { $_.Line -match 'Command Line:.*(-NullRHI|-RenderOffscreen|-ExecutePythonScript)' }).Count) { continue }
            $cacheLine = $cacheEvidence | Where-Object { $_.Line -match 'Using Local data cache path (.+?): Writable' } | Select-Object -Last 1
            if (!$cacheLine) { continue }
            if ($cacheLine.Line -match 'Using Local data cache path (.+?): Writable') {
                $candidateDataCachePath = $Matches[1].Trim().Trim('"')
                if (($candidateDataCachePath -match '^[A-Za-z]:[\\/]' -or $candidateDataCachePath -match '^[\\/]{2}[^\\/]+[\\/][^\\/]+') -and
                    (Test-Path -LiteralPath $candidateDataCachePath -PathType Container)) { $resolvedDataCachePath = $candidateDataCachePath }
            }
            break
        }
        catch { continue }
    }
}

if (![string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $resolvedDataCachePath, [EnvironmentVariableTarget]::Process)
}
# Run this launcher from the signed-in Windows desktop. Its process-only setting
# avoids rebuilding all mesh distance fields for this research inspection. The
# project config and the meshes' persistent build settings are left intact.
$editorArguments = @($Project, $result.new_level_path, '-nop4', '-nosourcecontrol',
    '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False')
$rendererScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'ue_render.py'
if (Test-Path -LiteralPath $rendererScript) {
    $workDirectory = Split-Path -Parent $AssemblyResult
    $viewportJobPath = Join-Path $workDirectory 'desktop_viewport_job.json'
    $viewportJob = [ordered]@{ assembly_result=$AssemblyResult; output_json=(Join-Path $workDirectory 'desktop_viewport_result.json'); mode='viewport'; inspection_daylight=$true; require_current_map=$true; quit_when_done=$false }
    [System.IO.File]::WriteAllText($viewportJobPath, ($viewportJob | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
    $env:UE_CITYCODE_RENDER_JOB = $viewportJobPath
    $editorArguments += '-ExecCmds=py "' + $rendererScript.Replace('\','/') + '"'
}
# A second whole-city Editor can exhaust both VRAM and RAM while the first is
# still preparing resources. Never close the user's current Editor to resolve it.
$existingEditor = @(Get-Process -Name UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue)
if ($existingEditor.Count -gt 0) {
    Write-Output 'A UE process is already running. No additional Editor was started.'
    if ((@($existingEditor | Where-Object { $_.ProcessName -eq 'UnrealEditor' }).Count -gt 0) -and (Test-Path -LiteralPath $rendererScript)) {
        Write-Output 'If it displays this CityCode map, apply the inspection sky in its console:'
        Write-Output ('py "' + $rendererScript.Replace('\','/') + '"')
    }
    Write-Output 'If the existing process is a background task, wait for it to finish. In a GUI, wait for shader, texture and mesh preparation before judging appearance.'
    # The .cmd wrapper pauses on this deferred-launch result so a double click
    # leaves the explanation visible instead of silently closing the window.
    exit 2
}
$quotedArguments = ($editorArguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' '
Start-Process -FilePath $EditorExe -ArgumentList $quotedArguments | Out-Null
Write-Output ('Opened CityCode scene: ' + $result.new_level_path)
Write-Output 'UE prepares shaders, textures, meshes and Lumen cards on first use. Wait for those preparation counters to finish before judging the appearance. Saved-map verification does not mean visual preparation is complete.'
