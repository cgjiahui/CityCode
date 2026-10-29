# Local wrapper regression only: fake delegates never start Unreal Editor.
# Run with Windows PowerShell 5.1 -NoProfile -ExecutionPolicy Bypass -File ...
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'Run-CityCode.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('citycode wrapper ' + [Guid]::NewGuid().ToString('N'))
$checks = 0
function Assert-True($Condition, [string]$Message) {
    if (!$Condition) { throw $Message }
    $script:checks += 1
}
function Write-Json([string]$Path, $Value) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 100), (New-Object Text.UTF8Encoding($false)))
}
function Assert-Throws($Action, [string]$Pattern) {
    $caught = $null
    try { & $Action | Out-Null }
    catch { $caught = $_.Exception.Message }
    Assert-True ($caught -and $caught -match $Pattern) ('Expected failure matching: ' + $Pattern + '; actual: ' + $caught)
}
try {
    $release = Join-Path $testRoot 'release with spaces'
    $adapter = Join-Path $release 'pipelines\ue_citycode'
    $engine = Join-Path $testRoot 'selected engine'
    $editor = Join-Path $engine 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
    $versionPath = Join-Path $engine 'Engine\Build\Build.version'
    $project = Join-Path $testRoot 'renamed project\My Renamed Project.uproject'
    $run = Join-Path $release 'build\new run'
    $cache = Join-Path $testRoot 'cache does not exist'
    New-Item -ItemType Directory -Force -Path $adapter, (Split-Path -Parent $editor) | Out-Null
    [IO.File]::WriteAllText($editor, 'Not an executable: this regression never invokes UE.')
    Write-Json $versionPath @{ MajorVersion=5; MinorVersion=0; PatchVersion=3; Changelist=20979098 }
    Write-Json $project @{ FileVersion=3; EngineAssociation='5.0'; Modules=@(@{ Name='CitySample'; Type='Runtime' }) }
    $delegate = @'
param($EditorExe, $Project, $WorkRoot, $LocalDataCachePath, $Action, $JobFile,
      $CoreLimit, [switch]$EnableRequiredPlugins, $Manifest, $NativeBridgeCatalog,
      $Workers, [switch]$ParallelWorkers)
$job = $null
if ($JobFile) { $job = Get-Content -Raw -Encoding UTF8 -LiteralPath $JobFile | ConvertFrom-Json }
[ordered]@{ fake_delegate=$true; action=$Action; job_file=$JobFile; job=$job
    manifest=$Manifest; native_catalog=$NativeBridgeCatalog; workers=$Workers
    parallel=[bool]$ParallelWorkers; enable_plugins=[bool]$EnableRequiredPlugins
    editor=$EditorExe; project=$Project; ddc=$LocalDataCachePath } | ConvertTo-Json -Depth 100
'@
    [IO.File]::WriteAllText((Join-Path $adapter 'invoke_editor.ps1'), $delegate)
    [IO.File]::WriteAllText((Join-Path $adapter 'Start-CityCode.ps1'), $delegate)
    $common = @{ EditorExe=$editor; Project=$project; LocalDataCachePath=$cache; ReleaseRoot=$release; RunRoot=$run }
    $priorCacheEnvironment = [Environment]::GetEnvironmentVariable('UE-LocalDataCachePath', [EnvironmentVariableTarget]::Process)
    $preview = (& $runner @common -Stage preflight -EnableRequiredPlugins | Out-String) | ConvertFrom-Json
    Assert-True (!$preview.execute -and $preview.compatibility.engine_version -eq '5.0.3') 'Default invocation must be a UE 5.0.3 plan.'
    Assert-True ($preview.compatibility.project_identity_evidence -contains 'module:CitySample') 'A renamed City Sample project must remain valid.'
    Assert-True (!(Test-Path -LiteralPath $run) -and !(Test-Path -LiteralPath $cache)) 'Dry run must not create run/DDC directories or job files.'
    Assert-True ([Environment]::GetEnvironmentVariable('UE-LocalDataCachePath', [EnvironmentVariableTarget]::Process) -ceq $priorCacheEnvironment) 'Dry run must not change the process DDC environment.'
    Assert-True (!$preview.plugins_will_be_enabled) 'Preview must not enable plugins.'
    $export = (& $runner @common -Stage export-bridge | Out-String) | ConvertFrom-Json
    Assert-True ($export.job.action -eq 'export_mesh_geometry' -and $export.job.mesh_paths.Count -eq 1) 'Export must preserve the explicit local mesh resource contract.'
    Assert-True ($export.job.geometry_output_directory -eq (Join-Path $release 'build\local_epic')) 'Default OBJ location must match the CPU workflow.'
    $import = (& $runner @common -Stage import-bridge | Out-String) | ConvertFrom-Json
    Assert-True ($import.missing_inputs -contains (Join-Path $run 'surfaces\joined\span_manifest.json')) 'Import preview must report a missing CPU template without starting UE.'
    Assert-True (!(Test-Path -LiteralPath $run)) 'All preview stages must remain read only.'
    Assert-Throws { & $runner @common -Stage preflight -Execute -DryRun } 'Choose -Execute or -DryRun'
    Write-Json $versionPath @{ MajorVersion=5; MinorVersion=1; PatchVersion=0 }
    Assert-Throws { & $runner @common -Stage preflight } 'verified with UE 5.0.3'
    Write-Json $versionPath @{ MajorVersion=5; MinorVersion=0; PatchVersion=3 }
    Write-Json $project @{ EngineAssociation='5.1'; Modules=@(@{ Name='CitySample' }) }
    Assert-Throws { & $runner @common -Stage preflight } 'does not match UE 5.0.3'
    Write-Json $project @{ EngineAssociation='5.0'; Modules=@(@{ Name='UnrelatedProject' }) }
    Assert-Throws { & $runner @common -Stage preflight } 'no CitySample module/plugin identity'
    Write-Json $project @{ EngineAssociation='5.0'; Modules=@(@{ Name='CitySample' }) }
    $preflight = (& $runner @common -Stage preflight -EnableRequiredPlugins -Execute | Out-String) | ConvertFrom-Json
    Assert-True ($preflight.fake_delegate -and $preflight.action -eq 'preflight' -and $preflight.enable_plugins) 'Execute must delegate explicit plugin authorization to the original preflight entry.'
    Assert-True ($preflight.job.output_json -eq (Join-Path $run 'windows\preflight_result.json')) 'The preflight job must use the derived output location.'
    $template = Join-Path $run 'surfaces\joined\span_manifest.json'
    Write-Json $template @{ templates=@() }
    $imported = (& $runner @common -Stage import-bridge -Execute | Out-String) | ConvertFrom-Json
    Assert-True ($imported.action -eq 'import_bridge_template' -and $imported.job.template_manifest -eq $template) 'Import must delegate the selected template manifest unchanged.'
    $manifest = Join-Path $run 'final\assembly_manifest.json'
    $native = Join-Path $run 'windows\native_bridge_catalog.json'
    Write-Json $manifest @{ schema='test_fixture' }
    Write-Json $native @{ assets=@() }
    $assembled = (& $runner @common -Stage assemble -Execute | Out-String) | ConvertFrom-Json
    Assert-True ($assembled.fake_delegate -and $assembled.manifest -eq $manifest -and $assembled.native_catalog -eq $native) 'Assembly must use Start-CityCode with derived manifest/catalog paths.'
    Assert-True ($assembled.workers -eq 4 -and !$assembled.parallel) 'Assembly must retain four logical shards with sequential execution by default.'
    Assert-True ($assembled.ddc -eq $cache -and $assembled.editor -eq $editor -and $assembled.project -eq $project) 'Explicit editor/project/DDC must reach the original entry unchanged.'
    [ordered]@{ status='passed'; checks=$checks; ue_started=$false; fake_delegates_only=$true } | ConvertTo-Json
}
finally {
    if (Test-Path -LiteralPath $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
