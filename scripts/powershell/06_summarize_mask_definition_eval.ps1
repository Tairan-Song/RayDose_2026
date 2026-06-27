param(
    [string]$StatsCsv = "data\photon\training\aux_mask_definition_eval\mask_definition_stats_even150.csv",
    [string]$SummaryCsv = "data\photon\training\aux_mask_definition_eval\mask_definition_summary_even150.csv"
)

$ErrorActionPreference = "Stop"

$stats = Import-Csv -LiteralPath $StatsCsv

function Get-Quantile {
    param([double[]]$Values, [double]$Q)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { return 0.0 }
    $idx = [Math]::Floor(($sorted.Count - 1) * $Q)
    return $sorted[$idx]
}

$summary = $stats |
    Group-Object definition |
    ForEach-Object {
        $vals = [double[]]@($_.Group | ForEach-Object { [double]$_.positive_percent })
        [PSCustomObject]@{
            definition = $_.Name
            files = $_.Count
            mean_positive_percent = ($vals | Measure-Object -Average).Average
            median_positive_percent = Get-Quantile $vals 0.50
            p05_positive_percent = Get-Quantile $vals 0.05
            p95_positive_percent = Get-Quantile $vals 0.95
            min_positive_percent = ($vals | Measure-Object -Minimum).Minimum
            max_positive_percent = ($vals | Measure-Object -Maximum).Maximum
        }
    } |
    Sort-Object mean_positive_percent -Descending

$summary | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation -Encoding UTF8
$summary
