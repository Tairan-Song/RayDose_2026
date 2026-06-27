param(
    [string]$TrainingDir = "data\photon\training",

    [double]$ThresholdFraction = 0.01,

    [string]$StatsCsv = "data\photon\training\dose_mask_stats_gt_1pct.csv",

    [int]$MaxFiles = 0,

    [int]$MaxDegreeOfParallelism = 4,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$source = @"
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

public sealed class DoseMaskBatchResult
{
    public int TotalFiles;
    public int ProcessedFiles;
    public int SkippedFiles;
    public long OutputBytes;
    public long PositiveVoxels;
    public long Voxels;
    public double MinPositivePercent = double.PositiveInfinity;
    public double MaxPositivePercent = double.NegativeInfinity;
    public double SumPositivePercent;
}

public static class PhotonDoseMaskBatch
{
    private sealed class MhaImage
    {
        public Dictionary<string, string> Meta;
        public byte[] RawBytes;
    }

    private sealed class FileStats
    {
        public string CaseId;
        public int BeamIdx;
        public int CpIdx;
        public string DosePath;
        public string MaskPath;
        public string DimSize;
        public string ElementSpacing;
        public string Offset;
        public double DoseMin;
        public double DoseMax;
        public double DoseMean;
        public long Voxels;
        public long DoseNonzeroVoxels;
        public long PositiveVoxels;
        public double PositivePercent;
        public double ThresholdValue;
        public long OutputBytes;
        public string Status;
    }

    public static DoseMaskBatchResult Process(
        string trainingDir,
        double thresholdFraction,
        string statsCsv,
        int maxFiles,
        int maxDegreeOfParallelism,
        bool force)
    {
        string fullTrainingDir = Path.GetFullPath(trainingDir);
        string fullStatsCsv = Path.GetFullPath(statsCsv);
        Directory.CreateDirectory(Path.GetDirectoryName(fullStatsCsv));

        var doseFiles = Directory.EnumerateFiles(fullTrainingDir, "Dose_B*_CP*.mha", SearchOption.AllDirectories)
            .Where(p => p.IndexOf(Path.DirectorySeparatorChar + "dose" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) >= 0)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        if (maxFiles > 0)
            doseFiles = doseFiles.Take(maxFiles).OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        var files = doseFiles.ToList();
        var result = new DoseMaskBatchResult { TotalFiles = files.Count };

        object writerLock = new object();
        object resultLock = new object();

        using (var writer = new StreamWriter(fullStatsCsv, false, new UTF8Encoding(false)))
        {
            writer.WriteLine("case_id,beam_idx,cp_idx,dose_path,mask_path,dim_size,element_spacing,offset,dose_min,dose_max,dose_mean,voxels,dose_nonzero_voxels,positive_voxels,positive_percent,threshold_value,output_bytes,status");

            var options = new ParallelOptions {
                MaxDegreeOfParallelism = Math.Max(1, maxDegreeOfParallelism)
            };

            Parallel.ForEach(files, options, dosePath =>
            {
                FileStats stats = null;
                try
                {
                    stats = ProcessOne(fullTrainingDir, dosePath, thresholdFraction, force);
                    if (stats.Status == "processed")
                    {
                        lock (resultLock)
                        {
                            result.ProcessedFiles++;
                            result.OutputBytes += stats.OutputBytes;
                            result.PositiveVoxels += stats.PositiveVoxels;
                            result.Voxels += stats.Voxels;
                            result.SumPositivePercent += stats.PositivePercent;
                            if (stats.PositivePercent < result.MinPositivePercent) result.MinPositivePercent = stats.PositivePercent;
                            if (stats.PositivePercent > result.MaxPositivePercent) result.MaxPositivePercent = stats.PositivePercent;
                        }
                    }
                    else
                    {
                        lock (resultLock)
                        {
                            result.SkippedFiles++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    stats = BuildPathStats(fullTrainingDir, dosePath);
                    stats.Status = "error: " + ex.Message.Replace("\r", " ").Replace("\n", " ");
                }

                lock (writerLock)
                {
                    writer.WriteLine(ToCsv(stats));
                }
            });
        }

        if (result.ProcessedFiles == 0)
        {
            result.MinPositivePercent = 0.0;
            result.MaxPositivePercent = 0.0;
        }

        return result;
    }

    private static FileStats ProcessOne(string trainingDir, string dosePath, double thresholdFraction, bool force)
    {
        FileStats pathStats = BuildPathStats(trainingDir, dosePath);
        if (File.Exists(pathStats.MaskPath) && !force)
        {
            pathStats.OutputBytes = new FileInfo(pathStats.MaskPath).Length;
            pathStats.Status = "skipped_existing";
            return pathStats;
        }

        MhaImage dose = ReadMha(dosePath);
        if (!dose.Meta.ContainsKey("ElementType") || dose.Meta["ElementType"] != "MET_FLOAT")
            throw new InvalidDataException("Expected MET_FLOAT dose");

        int voxels = checked(dose.RawBytes.Length / 4);
        byte[] mask = new byte[voxels];
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        double sum = 0.0;
        long nonzero = 0;

        for (int i = 0; i < voxels; i++)
        {
            float v = BitConverter.ToSingle(dose.RawBytes, i * 4);
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            if (Math.Abs(v) > 1e-12) nonzero++;
        }

        double threshold = max * thresholdFraction;
        long positive = 0;
        for (int i = 0; i < voxels; i++)
        {
            float v = BitConverter.ToSingle(dose.RawBytes, i * 4);
            if (v > threshold)
            {
                mask[i] = 1;
                positive++;
            }
        }

        byte[] compressedMask = Compress(mask);
        WriteMaskMha(dose.Meta, compressedMask, pathStats.MaskPath);

        var outInfo = new FileInfo(pathStats.MaskPath);
        pathStats.DimSize = GetMeta(dose.Meta, "DimSize");
        pathStats.ElementSpacing = GetMeta(dose.Meta, "ElementSpacing");
        pathStats.Offset = GetMeta(dose.Meta, "Offset");
        pathStats.DoseMin = min;
        pathStats.DoseMax = max;
        pathStats.DoseMean = sum / voxels;
        pathStats.Voxels = voxels;
        pathStats.DoseNonzeroVoxels = nonzero;
        pathStats.PositiveVoxels = positive;
        pathStats.PositivePercent = 100.0 * positive / voxels;
        pathStats.ThresholdValue = threshold;
        pathStats.OutputBytes = outInfo.Length;
        pathStats.Status = "processed";
        return pathStats;
    }

    private static FileStats BuildPathStats(string trainingDir, string dosePath)
    {
        string fullDosePath = Path.GetFullPath(dosePath);
        string caseDir = Directory.GetParent(Directory.GetParent(fullDosePath).FullName).FullName;
        string caseId = new DirectoryInfo(caseDir).Name;
        var match = Regex.Match(Path.GetFileName(fullDosePath), @"Dose_B(\d+)_CP(\d+)\.mha$", RegexOptions.IgnoreCase);
        int beamIdx = match.Success ? int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : -1;
        int cpIdx = match.Success ? int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture) : -1;
        string maskDir = Path.Combine(caseDir, "label_masks", "dose_gt_1pct");
        string stem = Path.GetFileNameWithoutExtension(fullDosePath);
        string maskPath = Path.Combine(maskDir, stem + "_mask.mha");
        return new FileStats
        {
            CaseId = caseId,
            BeamIdx = beamIdx,
            CpIdx = cpIdx,
            DosePath = fullDosePath,
            MaskPath = maskPath,
            Status = "pending"
        };
    }

    private static MhaImage ReadMha(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        string prefix = Encoding.ASCII.GetString(bytes, 0, Math.Min(bytes.Length, 16384));
        int marker = prefix.IndexOf("ElementDataFile = LOCAL", StringComparison.Ordinal);
        if (marker < 0)
            throw new InvalidDataException("Only LOCAL MHA files are supported");
        int lineEnd = prefix.IndexOf('\n', marker);
        if (lineEnd < 0)
            throw new InvalidDataException("Invalid MHA header");

        int headerLength = lineEnd + 1;
        string header = Encoding.ASCII.GetString(bytes, 0, headerLength);
        var meta = ParseHeader(header);

        byte[] payload = new byte[bytes.Length - headerLength];
        Buffer.BlockCopy(bytes, headerLength, payload, 0, payload.Length);

        bool compressed = meta.ContainsKey("CompressedData") &&
            meta["CompressedData"].Equals("True", StringComparison.OrdinalIgnoreCase);

        return new MhaImage
        {
            Meta = meta,
            RawBytes = compressed ? Decompress(payload) : payload
        };
    }

    private static Dictionary<string, string> ParseHeader(string header)
    {
        var meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string rawLine in header.Split('\n'))
        {
            string line = rawLine.Trim();
            if (line.Length == 0) continue;
            int eq = line.IndexOf('=');
            if (eq < 0) continue;
            meta[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
        }
        return meta;
    }

    private static byte[] Decompress(byte[] payload)
    {
        Exception last = null;
        foreach (int skip in new int[] { 0, 2 })
        {
            try
            {
                using (var input = new MemoryStream(payload))
                {
                    input.Position = skip;
                    using (var deflate = new DeflateStream(input, CompressionMode.Decompress))
                    using (var output = new MemoryStream())
                    {
                        deflate.CopyTo(output);
                        return output.ToArray();
                    }
                }
            }
            catch (Exception ex)
            {
                last = ex;
            }
        }
        throw new InvalidDataException("Could not decompress MHA payload", last);
    }

    private static byte[] Compress(byte[] raw)
    {
        using (var output = new MemoryStream())
        {
            using (var deflate = new DeflateStream(output, CompressionLevel.Optimal, true))
            {
                deflate.Write(raw, 0, raw.Length);
            }
            return output.ToArray();
        }
    }

    private static void WriteMaskMha(Dictionary<string, string> sourceMeta, byte[] compressedMask, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        var lines = new List<string>();
        foreach (string key in new string[] {
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
        })
        {
            if (sourceMeta.ContainsKey(key))
                lines.Add(key + " = " + sourceMeta[key]);
        }

        if (!sourceMeta.ContainsKey("BinaryData"))
            lines.Add("BinaryData = True");
        if (!sourceMeta.ContainsKey("BinaryDataByteOrderMSB"))
            lines.Add("BinaryDataByteOrderMSB = False");

        lines.Add("CompressedData = True");
        lines.Add("CompressedDataSize = " + compressedMask.Length.ToString(CultureInfo.InvariantCulture));
        lines.Add("ElementType = MET_UCHAR");
        lines.Add("ElementDataFile = LOCAL");
        string header = string.Join("\n", lines) + "\n";
        byte[] headerBytes = Encoding.ASCII.GetBytes(header);

        using (var fs = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            fs.Write(headerBytes, 0, headerBytes.Length);
            fs.Write(compressedMask, 0, compressedMask.Length);
        }
    }

    private static string GetMeta(Dictionary<string, string> meta, string key)
    {
        return meta.ContainsKey(key) ? meta[key] : "";
    }

    private static string ToCsv(FileStats s)
    {
        return string.Join(",", new string[] {
            Csv(s.CaseId),
            s.BeamIdx.ToString(CultureInfo.InvariantCulture),
            s.CpIdx.ToString(CultureInfo.InvariantCulture),
            Csv(s.DosePath),
            Csv(s.MaskPath),
            Csv(s.DimSize),
            Csv(s.ElementSpacing),
            Csv(s.Offset),
            D(s.DoseMin),
            D(s.DoseMax),
            D(s.DoseMean),
            s.Voxels.ToString(CultureInfo.InvariantCulture),
            s.DoseNonzeroVoxels.ToString(CultureInfo.InvariantCulture),
            s.PositiveVoxels.ToString(CultureInfo.InvariantCulture),
            D(s.PositivePercent),
            D(s.ThresholdValue),
            s.OutputBytes.ToString(CultureInfo.InvariantCulture),
            Csv(s.Status)
        });
    }

    private static string D(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value)) return "";
        return value.ToString("G17", CultureInfo.InvariantCulture);
    }

    private static string Csv(string value)
    {
        if (value == null) value = "";
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }
}
"@

Add-Type -TypeDefinition $source -Language CSharp

$training = Resolve-Path -LiteralPath $TrainingDir
$result = [PhotonDoseMaskBatch]::Process(
    $training.Path,
    $ThresholdFraction,
    (Join-Path (Get-Location) $StatsCsv),
    $MaxFiles,
    $MaxDegreeOfParallelism,
    [bool]$Force
)

[PSCustomObject]@{
    TrainingDir = $training.Path
    ThresholdFraction = $ThresholdFraction
    MaxDegreeOfParallelism = $MaxDegreeOfParallelism
    StatsCsv = (Resolve-Path -LiteralPath $StatsCsv).Path
    TotalFiles = $result.TotalFiles
    ProcessedFiles = $result.ProcessedFiles
    SkippedFiles = $result.SkippedFiles
    OutputGB = [Math]::Round($result.OutputBytes / 1GB, 3)
    OverallPositivePercent = if ($result.Voxels -gt 0) { [Math]::Round(100.0 * $result.PositiveVoxels / $result.Voxels, 6) } else { 0 }
    MeanPositivePercentPerFile = if ($result.ProcessedFiles -gt 0) { [Math]::Round($result.SumPositivePercent / $result.ProcessedFiles, 6) } else { 0 }
    MinPositivePercent = [Math]::Round($result.MinPositivePercent, 6)
    MaxPositivePercent = [Math]::Round($result.MaxPositivePercent, 6)
}
