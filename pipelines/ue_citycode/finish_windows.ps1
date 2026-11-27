param(
    [Parameter(Mandatory=$true)][string]$EditorExe,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][string]$WorkRoot,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string[]]$WorkerCheckpoints,
    [Parameter(Mandatory=$true)][string]$NativeBridgeCatalog,
    [string]$StatusFile,
    [int]$PollSeconds = 5
)
$ErrorActionPreference = 'Stop'
$adapterRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $adapterRoot 'invoke_editor.ps1'
if (!$StatusFile) { $StatusFile = Join-Path $WorkRoot 'task_status.json' }
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$manifestDoc = Get-Content -Raw -Encoding UTF8 -LiteralPath $Manifest | ConvertFrom-Json
$manifestDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLowerInvariant()
if ($manifestDoc.status -ne 'cpu_ready_ue_execution_pending') { throw 'Final frozen CPU manifest is not ready.' }
if (@($manifestDoc.diagnostic_subset_ids).Count -gt 0 -or !@($manifestDoc.meshes | Where-Object { $_.kind -eq 'background' }).Count) {
    throw 'The finish workflow requires the complete city and its prepared background.'
}
if (!@($manifestDoc.placements | Where-Object { $_.kind -eq 'predicted_bridge_span' }).Count -or
    !@($manifestDoc.placements | Where-Object { $_.placement_source -eq 'rule_based_road_semantics' }).Count) {
    throw 'The finish workflow requires the final native bridge spans and rule-based street props.'
}
$runId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + ([Guid]::NewGuid().ToString('N').Substring(0,8))
$assemblyOutput = Join-Path $WorkRoot 'assemble_result.json'
$readbackOutput = Join-Path $WorkRoot 'readback_result.json'

function Save-Json([string]$Path, $Value) {
    $temporary = $Path + '.writing'
    [System.IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
function Read-JsonIfReady([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) { return $null }
    try { return (Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json) }
    catch { return $null } # An import checkpoint may be between writes.
}
function Set-State([string]$Stage, [string]$Status, $Details) {
    Save-Json $StatusFile ([ordered]@{ schema='citycode.windows_finish_status.v1'; stage=$Stage; status=$Status;
        run_id=$runId; updated_utc=[DateTime]::UtcNow.ToString('o'); assembly_manifest=$Manifest;
        assembly_manifest_sha256=$manifestDigest; details=$Details })
}
function Assert-FrozenManifest {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLowerInvariant() -ne $manifestDigest) {
        throw 'Final assembly manifest changed while the workflow was running.'
    }
}
function Run-Stage([string]$Action, $Job) {
    Assert-FrozenManifest
    $jobFile = Join-Path $WorkRoot ('finish_' + $Action + '_job.json')
    Save-Json $jobFile $Job
    & $launcher -EditorExe $EditorExe -Project $Project -WorkRoot $WorkRoot -Action $Action -JobFile $jobFile | Out-Null
    $result = Read-JsonIfReady $Job.output_json
    if (!$result -or $result.status -eq 'failed') { throw "UE $Action did not produce a successful result." }
    return $result
}

try {
    # Validate completed worker *content hashes* against this final manifest.
    # The earlier worker scene/proposal ID may differ; immutable mesh IDs and
    # content hashes must still match every one of the final building inputs.
    $expectedImports = @{}
    foreach ($record in @($manifestDoc.meshes | Where-Object { $_.kind -in @('building','massing','surface') })) {
        if ($record.import_parts) { $parts = @($record.import_parts) }
        else { $parts = @([pscustomobject]@{sha256=$record.prepared.interchange_sha256}) }
        foreach ($part in $parts) { $expectedImports[$record.id + ':' + $part.sha256] = $true }
    }
    while ($true) {
        Assert-FrozenManifest
        $allReady = $true; $observed = @{}; $workerSummary = @()
        foreach ($checkpointPath in $WorkerCheckpoints) {
            $checkpoint = Read-JsonIfReady $checkpointPath
            if ($checkpoint -and $checkpoint.status -in @('failed','assets_import_stopped')) {
                throw "An import worker needs attention: $checkpointPath ($($checkpoint.status))"
            }
            if (!$checkpoint -or $checkpoint.status -ne 'assets_imported_complete') { $allReady = $false }
            if ($checkpoint) {
                $workerSummary += [pscustomobject]@{file=$checkpointPath;status=$checkpoint.status;imported=$checkpoint.imported_parts;total=$checkpoint.total_parts}
                foreach ($item in @($checkpoint.imports)) { $observed[$item.input_id + ':' + $item.sha256] = $true }
            }
        }
        $bridgeCatalog = Read-JsonIfReady $NativeBridgeCatalog
        if ($bridgeCatalog -and $bridgeCatalog.status -eq 'failed') { throw 'Native bridge import failed.' }
        if (!$bridgeCatalog -or $bridgeCatalog.status -ne 'native_bridge_template_imported' -or !$bridgeCatalog.bounds_verified -or !$bridgeCatalog.native_materials_verified) { $allReady = $false }
        Set-State 'waiting_for_asset_imports' 'running' @{workers=$workerSummary;bridge_catalog_ready=($null -ne $bridgeCatalog);matched_content_count=$observed.Count;expected_content_count=$expectedImports.Count}
        if ($allReady) {
            foreach ($key in $expectedImports.Keys) { if (!$observed.ContainsKey($key)) { throw "Worker outputs do not cover final building content: $key" } }
            $native = @{}
            foreach ($asset in @($bridgeCatalog.assets)) { $native[$asset.mesh_path] = $asset }
            foreach ($placement in @($manifestDoc.placements | Where-Object { $_.kind -eq 'predicted_bridge_span' })) {
                if (!$native.ContainsKey($placement.mesh_path) -or $native[$placement.mesh_path].template_sha256 -ne $placement.template_sha256) {
                    throw "Final native bridge content is not verified: $($placement.id)"
                }
            }
            break
        }
        Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min(60, $PollSeconds)))
    }
    Set-State 'importing_uploaded_background' 'running' @{checkpoint=(Join-Path $WorkRoot 'background_import_result.json')}
    $background = Run-Stage 'import_available' ([ordered]@{action='import_available';assembly_manifest=$Manifest;
        output_json=(Join-Path $WorkRoot 'background_import_result.json');include_kinds=@('background');
        worker_count=1;worker_index=0;wait_for_complete=$true;poll_seconds=$PollSeconds;
        require_distance_fields_disabled=$true;stop_file=(Join-Path $WorkRoot 'finish.stop')})
    if ($background.status -ne 'assets_imported_complete') { throw 'Background import did not complete.' }
    Set-State 'assembling_new_map' 'running' @{building_inputs=$expectedImports.Count;background_parts=$background.imported_parts}
    $assembly = Run-Stage 'assemble' ([ordered]@{action='assemble';assembly_manifest=$Manifest;output_json=$assemblyOutput;
        new_level_path=($manifestDoc.new_level_path + '_' + $runId);add_inspection_lighting=$true})
    if ($assembly.status -ne 'assembled_pending_independent_readback') { throw 'UE map assembly did not complete.' }
    Set-State 'independent_saved_map_readback' 'running' @{new_level_path=$assembly.new_level_path;assembly_result=$assemblyOutput}
    # A separate editor process reloads the saved map and saved asset packages.
    $verified = Run-Stage 'readback' ([ordered]@{action='readback';assembly_manifest=$Manifest;assembly_result=$assemblyOutput;
        output_json=$readbackOutput;bounds_tolerance_m=0.025;verify_inspection_lighting=$true;verify_generated_mesh_normals=$true})
    if ($verified.status -ne 'passed' -or !$verified.independent_saved_map_readback -or !$verified.inspection_lighting_checked -or !$verified.generated_normals_checked) { throw 'Independent saved-map geometry/lighting/normal verification failed.' }
    Assert-FrozenManifest
    Set-State 'complete' 'complete' @{new_level_path=$assembly.new_level_path;assembly_result=$assemblyOutput;
        readback_result=$readbackOutput;open_launcher=(Join-Path $adapterRoot 'Open-CityCode.ps1')}
    Get-Content -Raw -Encoding UTF8 -LiteralPath $StatusFile
}
catch {
    Set-State 'failed' 'failed' @{error=$_.Exception.Message;assembly_result=$assemblyOutput;readback_result=$readbackOutput}
    throw
}
