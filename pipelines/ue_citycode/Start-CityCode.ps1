param(
    [string]$EditorExe = 'D:\files\ue_assets\UE_5.0\Engine\Binaries\Win64\UnrealEditor-Cmd.exe',
    [string]$Project = 'D:\files\ue_assets\CitySample\CitySample.uproject',
    [string]$WorkRoot = 'D:\files\daily_tasks\pc-bridge-work\citycode',
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$NativeBridgeCatalog,
    [string]$LocalDataCachePath,
    [ValidateRange(1,16)][int]$Workers = 4,
    [switch]$ParallelWorkers,
    [switch]$DryRun,
    [switch]$ResumeOnly
)
$ErrorActionPreference = 'Stop'
$adapterRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryScript = Join-Path $adapterRoot 'ue_entry.py'
$finishScript = Join-Path $adapterRoot 'finish_windows.ps1'
$EditorExe = [System.IO.Path]::GetFullPath($EditorExe)
$Project = [System.IO.Path]::GetFullPath($Project)
$WorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)
$Manifest = [System.IO.Path]::GetFullPath($Manifest)
$NativeBridgeCatalog = [System.IO.Path]::GetFullPath($NativeBridgeCatalog)
$registryPath = Join-Path $WorkRoot 'citycode_workers.json'
$lockPath = Join-Path $WorkRoot 'citycode_start.lock'
$resolvedDataCachePath = $LocalDataCachePath
if ([string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    foreach ($environmentScope in @([EnvironmentVariableTarget]::Process, [EnvironmentVariableTarget]::User, [EnvironmentVariableTarget]::Machine)) {
        $candidateDataCachePath = [Environment]::GetEnvironmentVariable('UE-LocalDataCachePath', $environmentScope)
        if (![string]::IsNullOrWhiteSpace($candidateDataCachePath)) { $resolvedDataCachePath = $candidateDataCachePath; break }
    }
}
if ([string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
    $projectLogRoot = Join-Path ([IO.Path]::GetDirectoryName($Project)) 'Saved\Logs'
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

function Read-Json([string]$Path) {
    return (Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json)
}
function Read-Checkpoint([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try { return (Read-Json $Path) }
    catch { return $null } # The editor may currently be replacing its checkpoint.
}
function Save-Json([string]$Path, $Value) {
    $temporary = $Path + '.writing'
    [System.IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
function Get-ProcessRecord($Process) {
    return [pscustomobject]@{
        process_id = $Process.Id
        process_name = $Process.ProcessName
        start_time_utc = $Process.StartTime.ToUniversalTime().ToString('o')
        start_time_utc_ticks = $Process.StartTime.ToUniversalTime().Ticks.ToString()
    }
}
function Get-TrackedProcess($Record) {
    if (!$Record -or !$Record.process_id) { return $null }
    $tracked = Get-Process -Id ([int]$Record.process_id) -ErrorAction SilentlyContinue
    if (!$tracked) { return $null }
    # A PID can be reused. Never identify a process by its numeric ID alone.
    try {
        if ($tracked.ProcessName -ne $Record.process_name -or
            $tracked.StartTime.ToUniversalTime().Ticks.ToString() -ne $Record.start_time_utc_ticks) { return $null }
    }
    catch { throw "Cannot verify the identity of recorded process $($Record.process_id); refusing to control it." }
    return $tracked
}
function Quote-WindowsArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    # Start-Process joins its ArgumentList. Apply Windows argv quoting explicitly,
    # including backslashes preceding a quote or the closing quote.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}
function Assert-RegistryIdle($Registry) {
    if (!$Registry) { return }
    $activeController = Get-TrackedProcess $Registry.controller
    if ($activeController -and $Registry.controller_active -ne $false -and $activeController.Id -ne $PID) {
        throw "This WorkRoot already has a live CityCode controller (PID $($activeController.Id))."
    }
    if (!$ResumeOnly) {
        foreach ($worker in @($Registry.workers)) {
            $activeWorker = Get-TrackedProcess $worker
            if ($activeWorker) {
                throw "This WorkRoot already has a live CityCode worker (PID $($activeWorker.Id)). Use -ResumeOnly after its previous controller has exited."
            }
        }
    }
}
function Save-ControllerState([string]$Status, [string]$ErrorMessage = '') {
    $controllerState.status = $Status
    $controllerState.updated_utc = [DateTime]::UtcNow.ToString('o')
    $controllerState.error = $ErrorMessage
    Save-Json $registryPath $controllerState
}

foreach ($requiredFile in @($EditorExe, $Project, $Manifest, $NativeBridgeCatalog, $entryScript, $finishScript)) {
    if (!(Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Required file does not exist: $requiredFile" }
}
$manifestDoc = Read-Json $Manifest
$manifestDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLowerInvariant()
if ($manifestDoc.schema -ne 'citycode.ue_citycode_assembly.v1' -or $manifestDoc.status -ne 'cpu_ready_ue_execution_pending' -or
    @($manifestDoc.diagnostic_subset_ids).Count -gt 0) { throw 'A complete frozen CPU-ready assembly manifest is required.' }
if ($manifestDoc.counts.proposal_unresolved -ne 0 -or $manifestDoc.counts.interchange_pending -ne 0) {
    throw 'The manifest still has unresolved placements or unprepared interchange inputs.'
}
$buildingRecords = @($manifestDoc.meshes | Where-Object { $_.kind -in @('building','massing','surface') })
if (!$buildingRecords.Count -or !@($manifestDoc.meshes | Where-Object { $_.kind -eq 'background' }).Count) {
    throw 'The manifest must include building inputs and the prepared background.'
}
$nativeCatalog = Read-Json $NativeBridgeCatalog
if ($nativeCatalog.status -ne 'native_bridge_template_imported' -or !$nativeCatalog.bounds_verified -or !$nativeCatalog.native_materials_verified) {
    throw 'NativeBridgeCatalog must be the actual verified UE bridge import result.'
}
$nativeAssets = @{}
foreach ($asset in @($nativeCatalog.assets)) { $nativeAssets[$asset.mesh_path] = $asset }
$nativeSpans = @($manifestDoc.placements | Where-Object { $_.kind -eq 'predicted_bridge_span' })
if (!$nativeSpans.Count) { throw 'The manifest has no imported native predicted bridge spans.' }
foreach ($span in $nativeSpans) {
    if (!$nativeAssets.ContainsKey($span.mesh_path) -or $nativeAssets[$span.mesh_path].template_sha256 -ne $span.template_sha256) {
        throw "Native bridge catalog does not match the final span: $($span.id)"
    }
}

$previousRegistry = $null
if (Test-Path -LiteralPath $registryPath -PathType Leaf) { $previousRegistry = Read-Json $registryPath }
Assert-RegistryIdle $previousRegistry
if ($ResumeOnly) {
    if (!$previousRegistry -or $previousRegistry.schema -ne 'citycode.windows_workers.v1') { throw 'No compatible worker registry exists to resume.' }
    if ($previousRegistry.assembly_manifest_sha256 -ne $manifestDigest -or $previousRegistry.project -ne $Project) {
        throw 'Recorded workers belong to a different project or frozen manifest.'
    }
    if ($previousRegistry.status -eq 'complete') { throw 'This recorded run already completed; open its saved map instead.' }
    if ($PSBoundParameters.ContainsKey('Workers') -and $Workers -ne $previousRegistry.worker_count) { throw '-Workers differs from the recorded run.' }
    $Workers = [int]$previousRegistry.worker_count
    if (!$PSBoundParameters.ContainsKey('ParallelWorkers')) { $ParallelWorkers = [bool]$previousRegistry.parallel_workers }
    foreach ($worker in @($previousRegistry.workers)) {
        $checkpoint = Read-Checkpoint $worker.checkpoint
        if (!(Get-TrackedProcess $worker) -and (!$checkpoint -or $checkpoint.status -ne 'assets_imported_complete')) {
            $checkpoint = Read-Checkpoint $worker.checkpoint
            if (!$checkpoint -or $checkpoint.status -ne 'assets_imported_complete') {
                throw "Recorded worker is no longer running and has not completed: $($worker.worker_index)"
            }
        }
        if ($checkpoint -and $checkpoint.status -in @('failed','assets_import_stopped')) { throw 'A stopped or failed run requires a new normal invocation.' }
    }
}

$runId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + ([Guid]::NewGuid().ToString('N').Substring(0,8))
$runRoot = Join-Path (Join-Path $WorkRoot 'runs') $runId
$coreLimit = [Math]::Max(2, [int][Math]::Floor(16.0 / $Workers))
$plans = @()
if ($ResumeOnly) {
    $plans = @($previousRegistry.workers)
    $runId = $previousRegistry.run_id
    $runRoot = $previousRegistry.run_root
}
else {
    for ($workerIndex = 0; $workerIndex -lt $Workers; $workerIndex++) {
        $jobPath = Join-Path $runRoot ("import_w$workerIndex.job.json")
        $checkpointPath = Join-Path $runRoot ("import_w$workerIndex.json")
        $stopPath = Join-Path $runRoot ("import_w$workerIndex.stop")
        $logPath = Join-Path $runRoot ("import_w$workerIndex.log")
        $inputIds = @()
        for ($recordIndex = $workerIndex; $recordIndex -lt $buildingRecords.Count; $recordIndex += $Workers) {
            $inputIds += $buildingRecords[$recordIndex].id
        }
        $job = [ordered]@{action='import_available';assembly_manifest=$Manifest;output_json=$checkpointPath;
            include_kinds=@('building','massing','surface');worker_count=$Workers;worker_index=$workerIndex;
            wait_for_complete=$true;poll_seconds=5;require_distance_fields_disabled=$true;stop_file=$stopPath}
        $arguments = @($Project, '/Engine/Maps/Entry', '-unattended', '-nosplash', '-nosound', '-nop4', '-nosourcecontrol',
            '-EnablePlugins=GLTFImporter,EditorScriptingUtilities,PythonScriptPlugin', '-NullRHI',
            '-ini:Engine:[/Script/Engine.RendererSettings]:r.GenerateMeshDistanceFields=False',
            ('-corelimit=' + $coreLimit), ('-ExecutePythonScript=' + $entryScript), ('-abslog=' + $logPath))
        $plans += [pscustomobject]@{worker_index=$workerIndex;input_count=$inputIds.Count;input_ids=$inputIds;
            job_file=$jobPath;checkpoint=$checkpointPath;stop_file=$stopPath;log_file=$logPath;
            environment=@{UE_CITYCODE_JOB=$jobPath};job=$job;arguments=$arguments}
    }
}
$workerCheckpoints = @($plans | ForEach-Object { $_.checkpoint })
if ($DryRun) {
    [ordered]@{schema='citycode.windows_start_plan.v1';dry_run=$true;resume_only=[bool]$ResumeOnly;
        editor_exe=$EditorExe;project=$Project;work_root=$WorkRoot;assembly_manifest=$Manifest;
        assembly_manifest_sha256=$manifestDigest;worker_count=$Workers;worker_core_limit=$coreLimit;
        parallel_workers=[bool]$ParallelWorkers;max_concurrent_workers=$(if ($ParallelWorkers) {$Workers} else {1});
        local_data_cache_path=$resolvedDataCachePath;
        building_input_count=$buildingRecords.Count;native_bridge_count=$nativeSpans.Count;
        workers=$plans;finish_script=$finishScript;finish_worker_checkpoints=$workerCheckpoints;
        native_bridge_catalog=$NativeBridgeCatalog;ue_processes_started=0;files_written=0} | ConvertTo-Json -Depth 100
    return
}

$lockStream = $null; $finishPipeline = $null; $finishAsync = $null
$startedWorkers = @(); $launchedChildren = @(); $controllerState = $null; $succeeded = $false
try {
    New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
    try { $lockStream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None) }
    catch { throw 'Another CityCode controller holds this WorkRoot lock.' }
    # Close the check/start race after acquiring the exclusive controller lock.
    if (Test-Path -LiteralPath $registryPath) { Assert-RegistryIdle (Read-Json $registryPath) }
    if (![string]::IsNullOrWhiteSpace($resolvedDataCachePath)) {
        [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $resolvedDataCachePath, [EnvironmentVariableTarget]::Process)
    }
    $controllerState = [ordered]@{schema='citycode.windows_workers.v1';status='starting';run_id=$runId;run_root=$runRoot;
        controller=(Get-ProcessRecord (Get-Process -Id $PID));controller_active=$true;project=$Project;editor_exe=$EditorExe;
        assembly_manifest=$Manifest;assembly_manifest_sha256=$manifestDigest;native_bridge_catalog=$NativeBridgeCatalog;
        worker_count=$Workers;worker_core_limit=$coreLimit;parallel_workers=[bool]$ParallelWorkers;
        local_data_cache_path=$resolvedDataCachePath;
        workers=@();updated_utc='';error=''}
    if ($ResumeOnly) { $controllerState.workers = @($previousRegistry.workers) }
    else {
        New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
        Save-ControllerState 'starting'
        foreach ($plan in $plans) {
            Save-Json $plan.job_file $plan.job
            $oldJobEnvironment = $env:UE_CITYCODE_JOB
            try {
                $env:UE_CITYCODE_JOB = $plan.job_file
                $argumentLine = ($plan.arguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' '
                $child = Start-Process -FilePath $EditorExe -ArgumentList $argumentLine -PassThru
                # Retain the actual child handle before inspecting metadata, so
                # an exception while recording it cannot leave an orphan import.
                $launchedChildren += [pscustomobject]@{process=$child;stop_file=$plan.stop_file}
                $processRecord = Get-ProcessRecord $child
            }
            finally { $env:UE_CITYCODE_JOB = $oldJobEnvironment }
            $workerRecord = [pscustomobject]@{worker_index=$plan.worker_index;process_id=$processRecord.process_id;
                process_name=$processRecord.process_name;start_time_utc=$processRecord.start_time_utc;
                start_time_utc_ticks=$processRecord.start_time_utc_ticks;checkpoint=$plan.checkpoint;
                stop_file=$plan.stop_file;job_file=$plan.job_file;log_file=$plan.log_file;input_count=$plan.input_count}
            $startedWorkers += $workerRecord
            $controllerState.workers = @($startedWorkers)
            Save-ControllerState 'starting'
            if (!$ParallelWorkers) {
                Save-ControllerState 'importing_sequential'
                Write-Host ("CityCode shard {0}/{1} (PID {2}); waiting for its saved checkpoint." -f ($plan.worker_index+1), $Workers, $child.Id)
                while (!$child.HasExited) { Start-Sleep -Seconds 5 }
                $child.WaitForExit()
                if ($child.ExitCode -ne 0) {
                    throw "CityCode shard $($plan.worker_index) exited with code $($child.ExitCode). A checkpoint alone does not prove normal editor exit. Inspect $($plan.log_file)"
                }
                $finishedCheckpoint = Read-Checkpoint $plan.checkpoint
                if (!$finishedCheckpoint -or $finishedCheckpoint.status -ne 'assets_imported_complete') {
                    throw "CityCode shard $($plan.worker_index) exited without a complete checkpoint. Inspect $($plan.log_file)"
                }
            }
        }
    }
    if ($ParallelWorkers -and $launchedChildren.Count) {
        # Complete checkpoints can precede editor teardown. Validate every
        # owned process exit before finish_windows can publish task complete.
        Save-ControllerState 'waiting_for_worker_exit'
        foreach ($ownedChild in $launchedChildren) {
            $ownedChild.process.WaitForExit()
            if ($ownedChild.process.ExitCode -ne 0) {
                throw "A CityCode import worker exited with code $($ownedChild.process.ExitCode)."
            }
        }
    }
    $finishStop = Join-Path $WorkRoot 'finish.stop'
    if (Test-Path -LiteralPath $finishStop) { Remove-Item -LiteralPath $finishStop }
    Save-ControllerState 'running'
    Write-Host ("CityCode imports: {0} workers; status: {1}" -f $Workers, (Join-Path $WorkRoot 'task_status.json'))
    # The finish script runs in another runspace inside THIS PowerShell process.
    # The caller stays able to detect a crashed editor rather than waiting on a
    # partial checkpoint forever. No CIM access or extra controller process.
    $finishPipeline = [System.Management.Automation.PowerShell]::Create()
    [void]$finishPipeline.AddCommand($finishScript)
    [void]$finishPipeline.AddParameter('EditorExe',$EditorExe)
    [void]$finishPipeline.AddParameter('Project',$Project)
    [void]$finishPipeline.AddParameter('WorkRoot',$WorkRoot)
    [void]$finishPipeline.AddParameter('Manifest',$Manifest)
    [void]$finishPipeline.AddParameter('WorkerCheckpoints',[string[]]$workerCheckpoints)
    [void]$finishPipeline.AddParameter('NativeBridgeCatalog',$NativeBridgeCatalog)
    $finishAsync = $finishPipeline.BeginInvoke()
    while (!$finishAsync.IsCompleted) {
        $allWorkersComplete = $true
        foreach ($worker in @($controllerState.workers)) {
            $checkpoint = Read-Checkpoint $worker.checkpoint
            if (!$checkpoint -or $checkpoint.status -ne 'assets_imported_complete') {
                $allWorkersComplete = $false
                if (!(Get-TrackedProcess $worker)) {
                    # Re-read after observing exit: the previous read may have
                    # caught the editor's final checkpoint while it was open.
                    $checkpoint = Read-Checkpoint $worker.checkpoint
                    if (!$checkpoint -or $checkpoint.status -ne 'assets_imported_complete') {
                        throw "CityCode worker $($worker.worker_index) exited without a complete checkpoint. Inspect $($worker.log_file)"
                    }
                }
            }
            if ($checkpoint -and $checkpoint.status -in @('failed','assets_import_stopped')) {
                throw "CityCode worker $($worker.worker_index) needs attention: $($checkpoint.status)"
            }
        }
        if ($allWorkersComplete -and $controllerState.status -ne 'finishing') { Save-ControllerState 'finishing' }
        Start-Sleep -Seconds 5
    }
    $finishOutput = $finishPipeline.EndInvoke($finishAsync)
    if ($finishPipeline.HadErrors) { throw ($finishPipeline.Streams.Error | Out-String) }
    Save-ControllerState 'complete'
    $succeeded = $true
    $finishOutput | Write-Output
}
catch {
    if ($controllerState) { Save-ControllerState 'failed' $_.Exception.Message }
    throw
}
finally {
    if (!$succeeded -and $launchedChildren.Count) {
        # Only workers created by this invocation receive a stop request. They
        # finish and save their current complete part before inspecting it.
        foreach ($ownedChild in $launchedChildren) {
            [System.IO.File]::WriteAllText($ownedChild.stop_file, 'Stop after saving the current part.', (New-Object System.Text.UTF8Encoding($false)))
        }
        Write-Host 'Requested a safe stop after the current asset; waiting for this invocation''s workers.'
        do {
            $activeOwnedWorkers = @($launchedChildren | Where-Object { !$_.process.HasExited })
            if ($activeOwnedWorkers.Count) { Start-Sleep -Seconds 5 }
        } while ($activeOwnedWorkers.Count)
    }
    if ($finishPipeline) {
        if ($finishAsync -and !$finishAsync.IsCompleted) { $finishPipeline.Stop() }
        $finishPipeline.Dispose()
    }
    if ($controllerState -and $lockStream) {
        $controllerState.controller_active = $false
        try { Save-Json $registryPath $controllerState }
        catch { Write-Warning 'Could not persist controller exit; the exclusive WorkRoot lock is being released.' }
    }
    if ($lockStream) { $lockStream.Dispose() }
}
