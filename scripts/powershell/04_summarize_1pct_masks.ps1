param(
    [string]$StatsCsv = "data\photon\training\dose_mask_stats_gt_1pct.csv",
    [string]$PerCaseCsv = "data\photon\training\dose_mask_stats_gt_1pct_per_case.csv"
)

$ErrorActionPreference = "Stop"

$stats = Import-Csv -LiteralPath $StatsCsv
$caseRows = $stats | Group-Object case_id | ForEach-Object {
    $positive = $_.Group | ForEach-Object { [double]$_.positive_percent }
    $outputBytes = $_.Group | ForEach-Object { [int64]$_.output_bytes }
    [PSCustomObject]@{
        case_id = $_.Name
        files = $_.Count
        mean_positive_percent = ($positive | Measure-Object -Average).Average
        min_positive_percent = ($positive | Measure-Object -Minimum).Minimum
        max_positive_percent = ($positive | Measure-Object -Maximum).Maximum
        total_output_mb = (($outputBytes | Measure-Object -Sum).Sum / 1MB)
    }
} | Sort-Object case_id

$caseRows | Export-Csv -LiteralPath $PerCaseCsv -NoTypeInformation -Encoding UTF8

$positiveAll = $stats | ForEach-Object { [double]$_.positive_percent } | Sort-Object
function Get-Quantile {
    param([double[]]$Values, [double]$Q)
    $idx = [Math]::Floor(($Values.Count - 1) * $Q)
    return $Values[$idx]
}

[PSCustomObject]@{
    stats_csv = (Resolve-Path -LiteralPath $StatsCsv).Path
    per_case_csv = (Resolve-Path -LiteralPath $PerCaseCsv).Path
    files = $stats.Count
    cases = ($stats | Group-Object case_id).Count
    mean_positive_percent = ($positiveAll | Measure-Object -Average).Average
    median_positive_percent = Get-Quantile $positiveAll 0.50
    p05_positive_percent = Get-Quantile $positiveAll 0.05
    p95_positive_percent = Get-Quantile $positiveAll 0.95
    min_positive_percent = ($positiveAll | Measure-Object -Minimum).Minimum
    max_positive_percent = ($positiveAll | Measure-Object -Maximum).Maximum
}
