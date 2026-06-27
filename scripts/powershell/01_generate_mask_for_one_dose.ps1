param(
    [Parameter(Mandatory = $true)]
    [string]$DosePath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [double]$ThresholdFraction = 0.01,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Read-Mha {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = Resolve-Path -LiteralPath $Path
    $bytes = [System.IO.File]::ReadAllBytes($resolved.Path)
    $textPrefix = [System.Text.Encoding]::ASCII.GetString($bytes, 0, [Math]::Min($bytes.Length, 8192))
    $markerText = "ElementDataFile = LOCAL"
    $markerIndex = $textPrefix.IndexOf($markerText)
    if ($markerIndex -lt 0) {
        throw "Only LOCAL MHA files are supported: $Path"
    }

    $lineEnd = $textPrefix.IndexOf("`n", $markerIndex)
    if ($lineEnd -lt 0) {
        throw "Invalid MHA header: missing newline after ElementDataFile"
    }

    $headerLength = $lineEnd + 1
    $headerText = [System.Text.Encoding]::ASCII.GetString($bytes, 0, $headerLength)
    $payloadLength = $bytes.Length - $headerLength
    $payload = New-Object byte[] $payloadLength
    [Array]::Copy($bytes, $headerLength, $payload, 0, $payloadLength)

    $meta = [ordered]@{}
    foreach ($line in ($headerText -split "`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $meta[$parts[0].Trim()] = $parts[1].Trim()
    }

    $isCompressed = $false
    if ($meta.Contains("CompressedData")) {
        $isCompressed = ($meta["CompressedData"].ToLowerInvariant() -eq "true")
    }

    if ($isCompressed) {
        $raw = $null
        foreach ($skipBytes in @(0, 2)) {
            try {
                $ms = New-Object System.IO.MemoryStream(,$payload)
                if ($skipBytes -gt 0) {
                    $ms.Position = $skipBytes
                }
                $ds = New-Object System.IO.Compression.DeflateStream($ms, [System.IO.Compression.CompressionMode]::Decompress)
                $out = New-Object System.IO.MemoryStream
                $ds.CopyTo($out)
                $ds.Dispose()
                $raw = $out.ToArray()
                break
            }
            catch {
                $raw = $null
            }
        }

        if ($null -eq $raw) {
            throw "Could not decompress MHA payload: $Path"
        }
    }
    else {
        $raw = $payload
    }

    [PSCustomObject]@{
        Meta = $meta
        RawBytes = $raw
        HeaderText = $headerText
        Path = $resolved.Path
    }
}

function Get-FloatArrayStats {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)

    if (($Bytes.Length % 4) -ne 0) {
        throw "MET_FLOAT payload byte count is not divisible by 4"
    }

    $n = [int]($Bytes.Length / 4)
    $min = [double]::PositiveInfinity
    $max = [double]::NegativeInfinity
    $sum = 0.0
    $nonzero = 0

    for ($i = 0; $i -lt $n; $i++) {
        $v = [BitConverter]::ToSingle($Bytes, $i * 4)
        if ($v -lt $min) { $min = $v }
        if ($v -gt $max) { $max = $v }
        $sum += $v
        if ([Math]::Abs($v) -gt 1e-12) { $nonzero++ }
    }

    [PSCustomObject]@{
        Count = $n
        Min = $min
        Max = $max
        Mean = $sum / $n
        Nonzero = $nonzero
    }
}

function Write-MhaMask {
    param(
        [Parameter(Mandatory = $true)]$SourceMeta,
        [Parameter(Mandatory = $true)][byte[]]$MaskBytes,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $outDir = Split-Path -Parent $Path
    if ($outDir) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    $headerLines = New-Object System.Collections.Generic.List[string]
    foreach ($key in @(
        "ObjectType",
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "TransformMatrix",
        "Offset",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "DimSize"
    )) {
        if ($SourceMeta.Contains($key)) {
            $headerLines.Add("$key = $($SourceMeta[$key])")
        }
    }

    if (-not $SourceMeta.Contains("BinaryData")) {
        $headerLines.Add("BinaryData = True")
    }
    if (-not $SourceMeta.Contains("BinaryDataByteOrderMSB")) {
        $headerLines.Add("BinaryDataByteOrderMSB = False")
    }

    $headerLines.Add("CompressedData = False")
    $headerLines.Add("ElementType = MET_UCHAR")
    $headerLines.Add("ElementDataFile = LOCAL")
    $header = ($headerLines -join "`n") + "`n"
    $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)

    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        $fs.Write($headerBytes, 0, $headerBytes.Length)
        $fs.Write($MaskBytes, 0, $MaskBytes.Length)
    }
    finally {
        $fs.Dispose()
    }
}

if ((Test-Path -LiteralPath $OutputPath) -and -not $Force) {
    throw "Output already exists. Use -Force to overwrite: $OutputPath"
}

$mha = Read-Mha -Path $DosePath
if ($mha.Meta["ElementType"] -ne "MET_FLOAT") {
    throw "Expected dose ElementType MET_FLOAT, got $($mha.Meta['ElementType'])"
}

$stats = Get-FloatArrayStats -Bytes $mha.RawBytes
$threshold = $stats.Max * $ThresholdFraction
$mask = New-Object byte[] $stats.Count
$positive = 0

for ($i = 0; $i -lt $stats.Count; $i++) {
    $v = [BitConverter]::ToSingle($mha.RawBytes, $i * 4)
    if ($v -gt $threshold) {
        $mask[$i] = 1
        $positive++
    }
}

Write-MhaMask -SourceMeta $mha.Meta -MaskBytes $mask -Path $OutputPath

[PSCustomObject]@{
    DosePath = (Resolve-Path -LiteralPath $DosePath).Path
    OutputPath = (Resolve-Path -LiteralPath $OutputPath).Path
    ThresholdFraction = $ThresholdFraction
    DoseMax = $stats.Max
    ThresholdValue = $threshold
    Voxels = $stats.Count
    PositiveVoxels = $positive
    PositivePercent = [Math]::Round(100.0 * $positive / $stats.Count, 6)
    OutputElementType = "MET_UCHAR"
}
