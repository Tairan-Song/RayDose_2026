param(
    [string]$SummaryCsv = "data\photon\training\aux_mask_definition_eval\mask_definition_summary_full.csv",
    [string]$OutputCsv = "data\photon\training\aux_mask_definition_eval\mask_definition_recommendations.csv"
)

$ErrorActionPreference = "Stop"

$summary = Import-Csv -LiteralPath $SummaryCsv

function Get-Recommendation {
    param([string]$Definition)

    switch ($Definition) {
        "dose_gt_1pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 2
                dose_support_prediction_rank = 1
                edge_aware_rank = 0
                recommended_use = "Primary dose-support auxiliary label and two-head mask target"
                full_generation_priority = "Already generated full photon masks"
                caution = "Good default; may be slightly tight around penumbra"
            }
        }
        "dose_gt_1pct_dilate2" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 1
                dose_support_prediction_rank = 2
                edge_aware_rank = 0
                recommended_use = "Best candidate for weighted dose regression because it adds spatial margin"
                full_generation_priority = "Generate full masks after first training comparison"
                caution = "Dilation is voxel-space cube dilation; verify margin size vs spacing"
            }
        }
        "dose_gt_1pct_dilate1" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 3
                dose_support_prediction_rank = 3
                edge_aware_rank = 0
                recommended_use = "Conservative margin-expanded 1% support"
                full_generation_priority = "Optional"
                caution = "May be too close to original 1% mask to change training much"
            }
        }
        "dose_gt_0p5pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 4
                dose_support_prediction_rank = 4
                edge_aware_rank = 0
                recommended_use = "Broad low-dose context mask"
                full_generation_priority = "Optional comparison"
                caution = "Can include low-dose tail; broader than margin-expanded 1% masks"
            }
        }
        "dose_gt_2pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 5
                dose_support_prediction_rank = 5
                edge_aware_rank = 0
                recommended_use = "High-dose-region emphasis"
                full_generation_priority = "Optional"
                caution = "Too narrow for full dose-support supervision"
            }
        }
        "dose_gradient_top10pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 0
                dose_support_prediction_rank = 0
                edge_aware_rank = 1
                recommended_use = "Primary edge-aware loss candidate"
                full_generation_priority = "Stats sufficient first; generate if edge-aware training is selected"
                caution = "Boundary target, not a complete dose-support target"
            }
        }
        "dose_gradient_top5pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 0
                dose_support_prediction_rank = 0
                edge_aware_rank = 2
                recommended_use = "Tighter edge-aware loss candidate"
                full_generation_priority = "Optional"
                caution = "More selective; may underweight broader penumbra"
            }
        }
        "dose_gt_1pct_dilate3" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 6
                dose_support_prediction_rank = 6
                edge_aware_rank = 0
                recommended_use = "Broad margin-expanded support"
                full_generation_priority = "Low"
                caution = "Can become broad for large-open-aperture control points"
            }
        }
        "dose_gt_5pct" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 7
                dose_support_prediction_rank = 7
                edge_aware_rank = 0
                recommended_use = "High-dose core only"
                full_generation_priority = "Low"
                caution = "Too tight for main supervision"
            }
        }
        "dose_nonzero" {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 8
                dose_support_prediction_rank = 8
                edge_aware_rank = 0
                recommended_use = "Maximum-support baseline comparison"
                full_generation_priority = "Do not use as primary"
                caution = "Too broad; includes low-dose Monte Carlo tail/noise"
            }
        }
        default {
            return [PSCustomObject]@{
                dose_regression_weighting_rank = 99
                dose_support_prediction_rank = 99
                edge_aware_rank = 99
                recommended_use = "Unclassified"
                full_generation_priority = "Unknown"
                caution = ""
            }
        }
    }
}

$rows = foreach ($row in $summary) {
    $rec = Get-Recommendation -Definition $row.definition
    [PSCustomObject]@{
        definition = $row.definition
        files = [int]$row.files
        mean_positive_percent = [math]::Round([double]$row.mean_positive_percent, 4)
        median_positive_percent = [math]::Round([double]$row.median_positive_percent, 4)
        p05_positive_percent = [math]::Round([double]$row.p05_positive_percent, 4)
        p95_positive_percent = [math]::Round([double]$row.p95_positive_percent, 4)
        dose_regression_weighting_rank = $rec.dose_regression_weighting_rank
        dose_support_prediction_rank = $rec.dose_support_prediction_rank
        edge_aware_rank = $rec.edge_aware_rank
        recommended_use = $rec.recommended_use
        full_generation_priority = $rec.full_generation_priority
        caution = $rec.caution
    }
}

$rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
$rows | Sort-Object dose_regression_weighting_rank, edge_aware_rank
