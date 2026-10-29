param(
    [Parameter(Mandatory=$true)][string]$EditorExe,
    [string]$Project = 'D:\files\ue_assets\CitySample\CitySample.uproject',
    [string]$WorkRoot = 'D:\files\daily_tasks\pc-bridge-work\citycode',
    [ValidateSet('preflight','export_catalog','export_mesh_geometry','import_available','import_bridge_template','repair_missing_materials','persist_nanite_material_usage','repair_background_normals','configure_inspection_lighting','assemble','readback')][string]$Action = 'preflight',
    [string]$Manifest,
    [string]$AssemblyResult,
    [string]$JobFile,
    [string]$LocalDataCachePath,
    [switch]$EnableRequiredPlugins,
    [switch]$UseRendering,
    [switch]$BuildDistanceFields,
    [switch]$SkipShaderCompilation,
    [ValidateRange(0,64)][int]$CoreLimit = 0
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $EditorExe)) { throw "UE editor executable not found: $EditorExe" }
if (!(Test-Path -LiteralPath $Project)) { throw "UE project not found: $Project" }
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$adapterRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
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

# Enabling installed editor plugins changes only the project plugin list and
# keeps the exact original project descriptor in the dedicated work directory.
if ($EnableRequiredPlugins) {
    $projectDoc = Get-Content -Raw -Encoding UTF8 -LiteralPath $Project | ConvertFrom-Json
    $pluginList = @()
    if ($projectDoc.PSObject.Properties.Name -contains 'Plugins') { $pluginList = @($projectDoc.Plugins) }
    foreach ($pluginName in @('PythonScriptPlugin','EditorScriptingUtilities','GLTFImporter')) {
        $existingPlugin = @($pluginList | Where-Object { $_.Name -eq $pluginName })
        if ($existingPlugin.Count -gt 0) { $existingPlugin[0].Enabled = $true }
        else { $pluginList += [pscustomobject]@{ Name=$pluginName; Enabled=$true } }
    }
    $backupPath = Join-Path $WorkRoot ('CitySample.uproject.before_citycode_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
    Copy-Item -LiteralPath $Project -Destination $backupPath -ErrorAction Stop
    $projectDoc | Add-Member -Force -NotePropertyName Plugins -NotePropertyValue $pluginList
    [System.IO.File]::WriteAllText($Project, ($projectDoc | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))
}

if (!$JobFile) {
    $JobFile = Join-Path $WorkRoot ($Action + '_job.json')
    $job = [ordered]@{ action=$Action; output_json=(Join-Path $WorkRoot ($Action + '_result.json')) }
    if ($Action -eq 'assemble' -or $Action -eq 'readback' -or $Action -eq 'import_available') {
        if (!$Manifest) { throw '-Manifest is required for assemble/readback' }
        $job.assembly_manifest = $Manifest
    }
    if ($Action -eq 'readback') {
        if (!$AssemblyResult) { $AssemblyResult = Join-Path $WorkRoot 'assemble_result.json' }
        $job.assembly_result = $AssemblyResult
    }
    [System.IO.File]::WriteAllText($JobFile, ($job | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))
}
$jobDoc = Get-Content -Raw -Encoding UTF8 -LiteralPath $JobFile | ConvertFrom-Json
if (!$jobDoc.output_json) { throw 'Job JSON must contain output_json' }
# Remove a stale same-action result: absence of a fresh result after UE exits is
# an execution failure, never evidence that an old run succeeded.
if (Test-Path -LiteralPath $jobDoc.output_json) { Remove-Item -LiteralPath $jobDoc.output_json }
$env:UE_CITYCODE_JOB = (Resolve-Path -LiteralPath $JobFile).Path
$scriptPath = Join-Path $adapterRoot 'ue_entry.py'
if ($Action -eq 'readback') { $scriptPath = Join-Path $adapterRoot 'ue_readback.py' }
$logPath = Join-Path $WorkRoot ($Action + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$editorArguments = @($Project, '/Engine/Maps/Entry', '-unattended', '-nosplash', '-nosound', '-nop4', '-nosourcecontrol',
    '-EnablePlugins=GLTFImporter,EditorScriptingUtilities,PythonScriptPlugin',
    ('-ExecutePythonScript=' + $scriptPath), ('-abslog=' + $logPath))
if (!$UseRendering) { $editorArguments += '-NullRHI' }
if (!$BuildDistanceFields) { $editorArguments += '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False' }
if ($SkipShaderCompilation) { $editorArguments += '-NoShaderCompile' }
if ($CoreLimit -gt 0) { $editorArguments += ('-corelimit=' + $CoreLimit) }
if (![string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $resolvedDataCachePath, [EnvironmentVariableTarget]::Process)
}
# ExecutePythonScript uses the editor environment; commandlet mode is unsuitable
# for the editor-only level and GLTFImporter APIs used by this adapter.
& $EditorExe @editorArguments
$editorExit = $LASTEXITCODE
if (!(Test-Path -LiteralPath $jobDoc.output_json)) { throw "UE produced no result JSON (exit $editorExit). Inspect $logPath" }
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $jobDoc.output_json | ConvertFrom-Json
if ($editorExit -ne 0 -or $result.status -eq 'failed' -or ($Action -eq 'preflight' -and !$result.ready)) {
    throw "UE $Action failed. Inspect $($jobDoc.output_json) and $logPath"
}
$result | ConvertTo-Json -Depth 100
