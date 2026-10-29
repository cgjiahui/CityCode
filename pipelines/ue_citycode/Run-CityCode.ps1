param(
    [Parameter(Mandatory=$true)][ValidateSet('preflight','export-bridge','import-bridge','assemble')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$EditorExe,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][Alias('DDC')][ValidateNotNullOrEmpty()][string]$LocalDataCachePath,
    [string]$ReleaseRoot,
    [string]$RunRoot,
    [string]$GeometryOutputDirectory,
    [string[]]$BridgeMeshPath = @('/Game/Environment/Freeway/Mesh/SM_FREEWAY_VIS_349.SM_FREEWAY_VIS_349'),
    [string]$TemplateManifest,
    [string]$Manifest,
    [string]$NativeBridgeCatalog,
    [string]$NativeBridgeProposal,
    [ValidateRange(1,16)][int]$Workers = 4,
    [ValidateRange(1,64)][int]$CoreLimit = 8,
    [switch]$ParallelWorkers,
    [switch]$EnableRequiredPlugins,
    [switch]$DryRun,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
if ($Execute -and $DryRun) { throw 'Choose -Execute or -DryRun; the default is a plan without execution.' }
if ($EnableRequiredPlugins -and $Stage -ne 'preflight') { throw '-EnableRequiredPlugins is only used by the preflight stage.' }
if ($ParallelWorkers -and $Stage -ne 'assemble') { throw '-ParallelWorkers is only used by the assemble stage.' }
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
$ReleaseRoot = [IO.Path]::GetFullPath($ReleaseRoot)
if ([string]::IsNullOrWhiteSpace($RunRoot)) { $RunRoot = Join-Path $ReleaseRoot 'build\ue_citycode_replay' }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
$EditorExe = [IO.Path]::GetFullPath($EditorExe)
$Project = [IO.Path]::GetFullPath($Project)
if (![IO.Path]::IsPathRooted($LocalDataCachePath)) { throw '-LocalDataCachePath must be an absolute directory path.' }
$LocalDataCachePath = [IO.Path]::GetFullPath($LocalDataCachePath)
$adapterRoot = Join-Path $ReleaseRoot 'pipelines\ue_citycode'
$workRoot = Join-Path $RunRoot 'windows'
$invokeScript = Join-Path $adapterRoot 'invoke_editor.ps1'
$startScript = Join-Path $adapterRoot 'Start-CityCode.ps1'

function Read-Json([string]$Path) {
    return Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
}
function Normalize-Directory([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\','/'))
}
function Get-EditorCompatibility {
    if (!(Test-Path -LiteralPath $EditorExe -PathType Leaf)) { throw "UE editor executable not found: $EditorExe" }
    if ([IO.Path]::GetFileName($EditorExe) -notin @('UnrealEditor-Cmd.exe','UnrealEditor.exe')) {
        throw 'Use the UE 5.0.3 UnrealEditor-Cmd.exe or UnrealEditor.exe executable.'
    }
    $engineDirectory = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $EditorExe))
    $engineRoot = Split-Path -Parent $engineDirectory
    $versionPath = Join-Path $engineDirectory 'Build\Build.version'
    if (!(Test-Path -LiteralPath $versionPath -PathType Leaf)) { throw "UE Build.version not found: $versionPath" }
    $version = Read-Json $versionPath
    if ($version.MajorVersion -ne 5 -or $version.MinorVersion -ne 0 -or $version.PatchVersion -ne 3) {
        throw ('This pipeline was verified with UE 5.0.3; selected editor is {0}.{1}.{2}. Use the matching engine or validate another version separately.' -f $version.MajorVersion,$version.MinorVersion,$version.PatchVersion)
    }
    if (!(Test-Path -LiteralPath $Project -PathType Leaf) -or [IO.Path]::GetExtension($Project) -ne '.uproject') {
        throw "City Sample .uproject not found: $Project"
    }
    $projectDoc = Read-Json $Project
    $moduleNames = @($projectDoc.Modules | ForEach-Object { $_.Name })
    $pluginNames = @($projectDoc.Plugins | ForEach-Object { $_.Name })
    $identityEvidence = @($moduleNames | Where-Object { $_ -match '^CitySample($|Editor$|AnimGraphRuntime$)' } | ForEach-Object { 'module:' + $_ })
    $identityEvidence += @($pluginNames | Where-Object { $_ -match '^CitySample($|_)' } | ForEach-Object { 'plugin:' + $_ })
    if ($identityEvidence.Count -eq 0) {
        throw 'The project descriptor has no CitySample module/plugin identity. Its filename alone is not sufficient.'
    }
    $association = [string]$projectDoc.EngineAssociation
    $associationStatus = 'not_declared_selected_editor_verified'
    $registeredRoot = $null
    $warnings = @()
    if ($association -match '^(\d+)\.(\d+)(?:\.(\d+))?$') {
        if ([int]$Matches[1] -ne 5 -or [int]$Matches[2] -ne 0 -or ($Matches[3] -and [int]$Matches[3] -ne 3)) {
            throw "Project EngineAssociation '$association' does not match UE 5.0.3. Do not convert the project implicitly."
        }
        $associationStatus = 'numeric_compatible_selected_editor_verified'
    }
    elseif (![string]::IsNullOrWhiteSpace($association)) {
        # Read only this registered source-engine association, without dumping
        # other registry values or changing the project's engine association.
        $buildsKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Epic Games\Unreal Engine\Builds')
        if ($null -ne $buildsKey) {
            try { $registeredRoot = $buildsKey.GetValue($association, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) }
            finally { $buildsKey.Dispose() }
        }
        if (![string]::IsNullOrWhiteSpace([string]$registeredRoot)) {
            if ((Normalize-Directory ([string]$registeredRoot)) -ine (Normalize-Directory $engineRoot)) {
                throw "Project EngineAssociation '$association' is registered to a different engine: $registeredRoot"
            }
            $associationStatus = 'registered_engine_matches_selected_editor'
        }
        else {
            $associationStatus = 'unresolved_association_selected_editor_verified'
            $warnings += "EngineAssociation '$association' is not registered for this account; only the explicitly selected editor's Build.version was verified."
        }
    }
    return [ordered]@{
        engine_version='5.0.3'; build_version_file=$versionPath; changelist=$version.Changelist
        compatible_changelist=$version.CompatibleChangelist; branch=$version.BranchName
        engine_root=$engineRoot; project=$Project; project_identity_evidence=$identityEvidence
        project_engine_association=$association; association_status=$associationStatus
        registered_engine_root=$registeredRoot; warnings=$warnings
        runtime_preflight_required=$true
    }
}

# Planning reads descriptors and inputs only. In particular it does not create
# job files, set environment variables, modify plugins, or start any process.
$compatibility = Get-EditorCompatibility
if ([string]::IsNullOrWhiteSpace($GeometryOutputDirectory)) { $GeometryOutputDirectory = Join-Path $ReleaseRoot 'build\local_epic' }
if ([string]::IsNullOrWhiteSpace($TemplateManifest)) { $TemplateManifest = Join-Path $RunRoot 'surfaces\joined\span_manifest.json' }
if ([string]::IsNullOrWhiteSpace($Manifest)) { $Manifest = Join-Path $RunRoot 'final\assembly_manifest.json' }
if ([string]::IsNullOrWhiteSpace($NativeBridgeCatalog)) { $NativeBridgeCatalog = Join-Path $workRoot 'native_bridge_catalog.json' }
if ([string]::IsNullOrWhiteSpace($NativeBridgeProposal)) { $NativeBridgeProposal = Join-Path $workRoot 'native_bridge_proposal.json' }
$GeometryOutputDirectory = [IO.Path]::GetFullPath($GeometryOutputDirectory)
$TemplateManifest = [IO.Path]::GetFullPath($TemplateManifest)
$Manifest = [IO.Path]::GetFullPath($Manifest)
$NativeBridgeCatalog = [IO.Path]::GetFullPath($NativeBridgeCatalog)
$NativeBridgeProposal = [IO.Path]::GetFullPath($NativeBridgeProposal)
$job = $null
$jobFile = $null
$entryPoint = $invokeScript
$arguments = [ordered]@{ EditorExe=$EditorExe; Project=$Project; WorkRoot=$workRoot; LocalDataCachePath=$LocalDataCachePath }
$requiredInputs = @($invokeScript)
switch ($Stage) {
    'preflight' {
        $jobFile = Join-Path $workRoot 'preflight_job.json'
        $job = [ordered]@{ action='preflight'; output_json=(Join-Path $workRoot 'preflight_result.json') }
        $arguments.Action = 'preflight'
        if ($EnableRequiredPlugins) { $arguments.EnableRequiredPlugins = $true }
    }
    'export-bridge' {
        if ($BridgeMeshPath.Count -eq 0 -or @($BridgeMeshPath | Where-Object { $_ -notmatch '^/Game/.+\..+$' }).Count -gt 0) {
            throw '-BridgeMeshPath must contain explicit /Game/.../Mesh.Mesh resource paths.'
        }
        $jobFile = Join-Path $workRoot 'export_bridge_job.json'
        $job = [ordered]@{ action='export_mesh_geometry'; geometry_output_directory=$GeometryOutputDirectory
            mesh_paths=@($BridgeMeshPath); output_json=(Join-Path $GeometryOutputDirectory 'export_result.json') }
        $arguments.Action = 'export_mesh_geometry'
    }
    'import-bridge' {
        $requiredInputs += $TemplateManifest
        $jobFile = Join-Path $workRoot 'import_bridge_job.json'
        $job = [ordered]@{ action='import_bridge_template'; template_manifest=$TemplateManifest
            output_json=$NativeBridgeCatalog; output_proposal=$NativeBridgeProposal }
        $arguments.Action = 'import_bridge_template'
    }
    'assemble' {
        $entryPoint = $startScript
        $requiredInputs = @($startScript, $Manifest, $NativeBridgeCatalog)
        $arguments.Manifest = $Manifest
        $arguments.NativeBridgeCatalog = $NativeBridgeCatalog
        $arguments.Workers = $Workers
        if ($ParallelWorkers) { $arguments.ParallelWorkers = $true }
    }
}
if ($jobFile) { $arguments.JobFile = $jobFile; $arguments.CoreLimit = $CoreLimit }
$missingInputs = @($requiredInputs | Where-Object { !(Test-Path -LiteralPath $_ -PathType Leaf) })
$plan = [ordered]@{
    schema='citycode.windows_stage_plan.v1'; stage=$Stage; execute=[bool]$Execute
    release_root=$ReleaseRoot; run_root=$RunRoot; work_root=$workRoot
    local_data_cache_path=$LocalDataCachePath; local_data_cache_exists=(Test-Path -LiteralPath $LocalDataCachePath -PathType Container)
    compatibility=$compatibility; entry_point=$entryPoint; parameters=$arguments
    job_file=$jobFile; job=$job; missing_inputs=$missingInputs
    plugins_will_be_enabled=([bool]$Execute -and [bool]$EnableRequiredPlugins)
    worker_concurrency=$(if ($Stage -eq 'assemble') { if ($ParallelWorkers) { $Workers } else { 1 } } else { 1 })
    next_step=$(if ($Execute) { 'Delegating to the existing verified entry point.' } else { 'Review this plan, then repeat the same command with -Execute.' })
}
if (!$Execute) { $plan | ConvertTo-Json -Depth 100; return }
if ($missingInputs.Count -gt 0) { throw ('Missing stage input(s): ' + ($missingInputs -join ', ')) }
if ($jobFile) {
    New-Item -ItemType Directory -Force -Path $workRoot | Out-Null
    foreach ($outputPath in @($job.output_json, $job.output_proposal)) {
        if ($outputPath) { New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null }
    }
    [IO.File]::WriteAllText($jobFile, ($job | ConvertTo-Json -Depth 100), (New-Object Text.UTF8Encoding($false)))
}
# The delegated scripts own UE startup, result/exit checks, process-only DDC,
# plugin backups, shard scheduling and independent saved-map verification.
& $entryPoint @arguments
