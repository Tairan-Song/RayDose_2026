param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir,

    [double]$ThresholdFraction = 0.01,

    [int]$MaxFiles = 0,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$casePath = Resolve-Path -LiteralPath $CaseDir
$doseDir = Join-Path $casePath.Path "dose"
if (-not (Test-Path -LiteralPath $doseDir)) {
    throw "Dose directory not found: $doseDir"
}

if ($MaxFiles -le 0) {
    throw "Refusing to process an unlimited number of dose files. Pass -MaxFiles explicitly."
}

$thresholdName = ("dose_gt_{0:p0}" -f $ThresholdFraction).Replace("%", "pct").Replace(" ", "")
$outDir = Join-Path $casePath.Path "label_masks\$thresholdName"
$scriptPath = Join-Path $PSScriptRoot "01_generate_mask_for_one_dose.ps1"

$doseFiles = Get-ChildItem -LiteralPath $doseDir -Filter "Dose_B*_CP*.mha" -File |
    Sort-Object Name |
    Select-Object -First $MaxFiles

$results = New-Object System.Collections.Generic.List[object]

foreach ($doseFile in $doseFiles) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($doseFile.Name)
    $outPath = Join-Path $outDir "$($stem)_mask.mha"
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $scriptPath,
        "-DosePath", $doseFile.FullName,
        "-OutputPath", $outPath,
        "-ThresholdFraction", $ThresholdFraction
    )
    if ($Force) {
        $args += "-Force"
    }

    $result = & powershell @args
    $results.Add($result)
}

[PSCustomObject]@{
    CaseDir = $casePath.Path
    OutputDir = $outDir
    ThresholdFraction = $ThresholdFraction
    RequestedMaxFiles = $MaxFiles
    ProcessedFiles = $doseFiles.Count
}
