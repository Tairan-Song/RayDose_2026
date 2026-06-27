param(
    [string]$TrainingDir = "data\photon\training",

    [string]$OutputDir = "data\photon\training\aux_mask_definition_eval",

    [string]$StatsCsv = "data\photon\training\aux_mask_definition_eval\mask_definition_stats.csv",

    [int]$MaxFiles = 30,

    [ValidateSet("First", "Even")]
    [string]$SampleMode = "Even",

    [int]$MaxDegreeOfParallelism = 2,

    [switch]$WriteMasks,

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

public sealed class AuxMaskEvalResult
{
    public int DoseFiles;
    public int Rows;
    public int ProcessedRows;
    public int ErrorRows;
    public long OutputBytes;
}

public static class AuxMaskDefinitionEval
{
    private sealed class MhaImage
    {
        public Dictionary<string, string> Meta;
        public byte[] RawBytes;
    }

    private sealed class DoseInfo
    {
        public string CaseId;
        public int BeamIdx;
        public int CpIdx;
        public string DosePath;
    }

    private sealed class Row
    {
        public string CaseId;
        public int BeamIdx;
        public int CpIdx;
        public string Definition;
        public string DosePath;
        public string MaskPath;
        public string DimSize;
        public string ElementSpacing;
        public double DoseMax;
        public double ThresholdValue;
        public int DilationVoxels;
        public long Voxels;
        public long PositiveVoxels;
        public double PositivePercent;
        public long OutputBytes;
        public string Status;
        public string Notes;
    }

    public static AuxMaskEvalResult Process(
        string trainingDir,
        string outputDir,
        string statsCsv,
        int maxFiles,
        string sampleMode,
        int maxDegreeOfParallelism,
        bool writeMasks,
        bool force)
    {
        string fullTrainingDir = Path.GetFullPath(trainingDir);
        string fullOutputDir = Path.GetFullPath(outputDir);
        string fullStatsCsv = Path.GetFullPath(statsCsv);
        Directory.CreateDirectory(fullOutputDir);
        Directory.CreateDirectory(Path.GetDirectoryName(fullStatsCsv));

        var allFiles = Directory.EnumerateFiles(fullTrainingDir, "Dose_B*_CP*.mha", SearchOption.AllDirectories)
            .Where(p => p.IndexOf(Path.DirectorySeparatorChar + "dose" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) >= 0)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var doseFiles = SelectFiles(allFiles, maxFiles, sampleMode);
        var result = new AuxMaskEvalResult { DoseFiles = doseFiles.Count };
        object writerLock = new object();
        object resultLock = new object();

        using (var writer = new StreamWriter(fullStatsCsv, false, new UTF8Encoding(false)))
        {
            writer.WriteLine("case_id,beam_idx,cp_idx,definition,dose_path,mask_path,dim_size,element_spacing,dose_max,threshold_value,dilation_voxels,voxels,positive_voxels,positive_percent,output_bytes,status,notes");
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, maxDegreeOfParallelism) };

            Parallel.ForEach(doseFiles, options, dosePath =>
            {
                List<Row> rows;
                try
                {
                    rows = ProcessDose(fullTrainingDir, fullOutputDir, dosePath, writeMasks, force);
                }
                catch (Exception ex)
                {
                    var info = BuildDoseInfo(dosePath);
                    rows = new List<Row> {
                        new Row {
                            CaseId = info.CaseId,
                            BeamIdx = info.BeamIdx,
                            CpIdx = info.CpIdx,
                            Definition = "all",
                            DosePath = info.DosePath,
                            Status = "error",
                            Notes = ex.Message.Replace("\r", " ").Replace("\n", " ")
                        }
                    };
                }

                lock (writerLock)
                {
                    foreach (var row in rows)
                        writer.WriteLine(ToCsv(row));
                }

                lock (resultLock)
                {
                    result.Rows += rows.Count;
                    result.ProcessedRows += rows.Count(r => r.Status == "processed" || r.Status == "stats_only");
                    result.ErrorRows += rows.Count(r => r.Status == "error");
                    result.OutputBytes += rows.Sum(r => r.OutputBytes);
                }
            });
        }

        return result;
    }

    private static List<string> SelectFiles(List<string> allFiles, int maxFiles, string sampleMode)
    {
        if (maxFiles <= 0 || maxFiles >= allFiles.Count)
            return allFiles;
        if (sampleMode.Equals("First", StringComparison.OrdinalIgnoreCase))
            return allFiles.Take(maxFiles).ToList();

        var selected = new List<string>();
        if (maxFiles == 1)
        {
            selected.Add(allFiles[0]);
            return selected;
        }

        double step = (allFiles.Count - 1) / (double)(maxFiles - 1);
        var used = new HashSet<int>();
        for (int i = 0; i < maxFiles; i++)
        {
            int idx = (int)Math.Round(i * step);
            idx = Math.Max(0, Math.Min(allFiles.Count - 1, idx));
            if (used.Add(idx))
                selected.Add(allFiles[idx]);
        }
        return selected;
    }

    private static List<Row> ProcessDose(string trainingDir, string outputDir, string dosePath, bool writeMasks, bool force)
    {
        var info = BuildDoseInfo(dosePath);
        var mha = ReadMha(dosePath);
        if (!mha.Meta.ContainsKey("ElementType") || mha.Meta["ElementType"] != "MET_FLOAT")
            throw new InvalidDataException("Expected MET_FLOAT dose");

        int[] dims = ParseDims(GetMeta(mha.Meta, "DimSize"));
        int voxels = checked(dims[0] * dims[1] * dims[2]);
        if (mha.RawBytes.Length != voxels * 4)
            throw new InvalidDataException("Payload size does not match DimSize");

        float[] dose = new float[voxels];
        double max = double.NegativeInfinity;
        for (int i = 0; i < voxels; i++)
        {
            float v = BitConverter.ToSingle(mha.RawBytes, i * 4);
            dose[i] = v;
            if (v > max) max = v;
        }

        var rows = new List<Row>();
        AddMaskRow(rows, info, mha.Meta, outputDir, "dose_nonzero", dose, dims, max, 0.0, 0, v => v > 0f, writeMasks, force, "mask = dose > 0");
        AddMaskRow(rows, info, mha.Meta, outputDir, "dose_gt_0p5pct", dose, dims, max, 0.005 * max, 0, v => v > 0.005 * max, writeMasks, force, "mask = dose > 0.5% max dose");
        AddMaskRow(rows, info, mha.Meta, outputDir, "dose_gt_1pct", dose, dims, max, 0.01 * max, 0, v => v > 0.01 * max, writeMasks, force, "mask = dose > 1% max dose");
        AddMaskRow(rows, info, mha.Meta, outputDir, "dose_gt_2pct", dose, dims, max, 0.02 * max, 0, v => v > 0.02 * max, writeMasks, force, "mask = dose > 2% max dose");
        AddMaskRow(rows, info, mha.Meta, outputDir, "dose_gt_5pct", dose, dims, max, 0.05 * max, 0, v => v > 0.05 * max, writeMasks, force, "mask = dose > 5% max dose");

        byte[] base1Pct = BuildMask(dose, v => v > 0.01 * max);
        AddDilationRow(rows, info, mha.Meta, outputDir, "dose_gt_1pct_dilate1", base1Pct, dims, max, 0.01 * max, 1, writeMasks, force);
        AddDilationRow(rows, info, mha.Meta, outputDir, "dose_gt_1pct_dilate2", base1Pct, dims, max, 0.01 * max, 2, writeMasks, force);
        AddDilationRow(rows, info, mha.Meta, outputDir, "dose_gt_1pct_dilate3", base1Pct, dims, max, 0.01 * max, 3, writeMasks, force);

        AddGradientRows(rows, info, mha.Meta, outputDir, dose, dims, max, writeMasks, force);
        return rows;
    }

    private static void AddMaskRow(
        List<Row> rows,
        DoseInfo info,
        Dictionary<string, string> meta,
        string outputDir,
        string definition,
        float[] dose,
        int[] dims,
        double doseMax,
        double threshold,
        int dilationVoxels,
        Func<float, bool> predicate,
        bool writeMasks,
        bool force,
        string notes)
    {
        byte[] mask = BuildMask(dose, predicate);
        AddByteMaskRow(rows, info, meta, outputDir, definition, mask, dims, doseMax, threshold, dilationVoxels, writeMasks, force, notes);
    }

    private static void AddDilationRow(
        List<Row> rows,
        DoseInfo info,
        Dictionary<string, string> meta,
        string outputDir,
        string definition,
        byte[] baseMask,
        int[] dims,
        double doseMax,
        double threshold,
        int radius,
        bool writeMasks,
        bool force)
    {
        byte[] dilated = DilateCube(baseMask, dims, radius);
        AddByteMaskRow(rows, info, meta, outputDir, definition, dilated, dims, doseMax, threshold, radius, writeMasks, force, "26-neighborhood cube dilation of 1% mask by " + radius.ToString(CultureInfo.InvariantCulture) + " voxel(s)");
    }

    private static void AddGradientRows(
        List<Row> rows,
        DoseInfo info,
        Dictionary<string, string> meta,
        string outputDir,
        float[] dose,
        int[] dims,
        double doseMax,
        bool writeMasks,
        bool force)
    {
        int nx = dims[0], ny = dims[1], nz = dims[2];
        float[] grad = new float[dose.Length];
        var nonzero = new List<float>();

        for (int z = 1; z < nz - 1; z++)
        {
            int zOff = z * nx * ny;
            for (int y = 1; y < ny - 1; y++)
            {
                int off = zOff + y * nx;
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = off + x;
                    float gx = 0.5f * (dose[idx + 1] - dose[idx - 1]);
                    float gy = 0.5f * (dose[idx + nx] - dose[idx - nx]);
                    float gz = 0.5f * (dose[idx + nx * ny] - dose[idx - nx * ny]);
                    float g = (float)Math.Sqrt(gx * gx + gy * gy + gz * gz);
                    grad[idx] = g;
                    if (g > 0f) nonzero.Add(g);
                }
            }
        }

        if (nonzero.Count == 0)
        {
            AddByteMaskRow(rows, info, meta, outputDir, "dose_gradient_top10pct", new byte[dose.Length], dims, doseMax, 0, 0, writeMasks, force, "empty gradient");
            AddByteMaskRow(rows, info, meta, outputDir, "dose_gradient_top5pct", new byte[dose.Length], dims, doseMax, 0, 0, writeMasks, force, "empty gradient");
            return;
        }

        nonzero.Sort();
        double thr10 = Quantile(nonzero, 0.90);
        double thr05 = Quantile(nonzero, 0.95);
        AddByteMaskRow(rows, info, meta, outputDir, "dose_gradient_top10pct", BuildMask(grad, v => v > thr10), dims, doseMax, thr10, 0, writeMasks, force, "mask = top 10% nonzero |gradient(dose)| voxels");
        AddByteMaskRow(rows, info, meta, outputDir, "dose_gradient_top5pct", BuildMask(grad, v => v > thr05), dims, doseMax, thr05, 0, writeMasks, force, "mask = top 5% nonzero |gradient(dose)| voxels");
    }

    private static void AddByteMaskRow(
        List<Row> rows,
        DoseInfo info,
        Dictionary<string, string> meta,
        string outputDir,
        string definition,
        byte[] mask,
        int[] dims,
        double doseMax,
        double threshold,
        int dilationVoxels,
        bool writeMasks,
        bool force,
        string notes)
    {
        long positive = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i] != 0) positive++;

        string maskPath = "";
        long outputBytes = 0;
        string status = "stats_only";
        if (writeMasks)
        {
            maskPath = Path.Combine(outputDir, definition, info.CaseId, String.Format(CultureInfo.InvariantCulture, "Dose_B{0}_CP{1:000}_mask.mha", info.BeamIdx, info.CpIdx));
            if (File.Exists(maskPath) && !force)
            {
                outputBytes = new FileInfo(maskPath).Length;
                status = "skipped_existing";
            }
            else
            {
                byte[] compressed = Compress(mask);
                WriteMaskMha(meta, compressed, maskPath);
                outputBytes = new FileInfo(maskPath).Length;
                status = "processed";
            }
        }

        rows.Add(new Row {
            CaseId = info.CaseId,
            BeamIdx = info.BeamIdx,
            CpIdx = info.CpIdx,
            Definition = definition,
            DosePath = info.DosePath,
            MaskPath = maskPath,
            DimSize = GetMeta(meta, "DimSize"),
            ElementSpacing = GetMeta(meta, "ElementSpacing"),
            DoseMax = doseMax,
            ThresholdValue = threshold,
            DilationVoxels = dilationVoxels,
            Voxels = mask.Length,
            PositiveVoxels = positive,
            PositivePercent = 100.0 * positive / mask.Length,
            OutputBytes = outputBytes,
            Status = status,
            Notes = notes
        });
    }

    private static byte[] BuildMask(float[] values, Func<float, bool> predicate)
    {
        byte[] mask = new byte[values.Length];
        for (int i = 0; i < values.Length; i++)
            if (predicate(values[i])) mask[i] = 1;
        return mask;
    }

    private static byte[] DilateCube(byte[] mask, int[] dims, int radius)
    {
        int nx = dims[0], ny = dims[1], nz = dims[2];
        byte[] output = new byte[mask.Length];
        int plane = nx * ny;

        for (int z = 0; z < nz; z++)
        {
            for (int y = 0; y < ny; y++)
            {
                int off = z * plane + y * nx;
                for (int x = 0; x < nx; x++)
                {
                    int idx = off + x;
                    if (mask[idx] == 0) continue;
                    int z0 = Math.Max(0, z - radius), z1 = Math.Min(nz - 1, z + radius);
                    int y0 = Math.Max(0, y - radius), y1 = Math.Min(ny - 1, y + radius);
                    int x0 = Math.Max(0, x - radius), x1 = Math.Min(nx - 1, x + radius);
                    for (int zz = z0; zz <= z1; zz++)
                        for (int yy = y0; yy <= y1; yy++)
                        {
                            int oo = zz * plane + yy * nx;
                            for (int xx = x0; xx <= x1; xx++)
                                output[oo + xx] = 1;
                        }
                }
            }
        }
        return output;
    }

    private static double Quantile(List<float> sorted, double q)
    {
        int idx = (int)Math.Floor((sorted.Count - 1) * q);
        return sorted[Math.Max(0, Math.Min(sorted.Count - 1, idx))];
    }

    private static DoseInfo BuildDoseInfo(string dosePath)
    {
        string fullDosePath = Path.GetFullPath(dosePath);
        string caseDir = Directory.GetParent(Directory.GetParent(fullDosePath).FullName).FullName;
        string caseId = new DirectoryInfo(caseDir).Name;
        var match = Regex.Match(Path.GetFileName(fullDosePath), @"Dose_B(\d+)_CP(\d+)\.mha$", RegexOptions.IgnoreCase);
        return new DoseInfo {
            CaseId = caseId,
            BeamIdx = match.Success ? int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : -1,
            CpIdx = match.Success ? int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture) : -1,
            DosePath = fullDosePath
        };
    }

    private static MhaImage ReadMha(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        string prefix = Encoding.ASCII.GetString(bytes, 0, Math.Min(bytes.Length, 16384));
        int marker = prefix.IndexOf("ElementDataFile = LOCAL", StringComparison.Ordinal);
        if (marker < 0) throw new InvalidDataException("Only LOCAL MHA files are supported");
        int lineEnd = prefix.IndexOf('\n', marker);
        if (lineEnd < 0) throw new InvalidDataException("Invalid MHA header");

        int headerLength = lineEnd + 1;
        var meta = ParseHeader(Encoding.ASCII.GetString(bytes, 0, headerLength));
        byte[] payload = new byte[bytes.Length - headerLength];
        Buffer.BlockCopy(bytes, headerLength, payload, 0, payload.Length);
        bool compressed = meta.ContainsKey("CompressedData") && meta["CompressedData"].Equals("True", StringComparison.OrdinalIgnoreCase);
        return new MhaImage { Meta = meta, RawBytes = compressed ? Decompress(payload) : payload };
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
            catch (Exception ex) { last = ex; }
        }
        throw new InvalidDataException("Could not decompress MHA payload", last);
    }

    private static byte[] Compress(byte[] raw)
    {
        using (var output = new MemoryStream())
        {
            using (var deflate = new DeflateStream(output, CompressionLevel.Optimal, true))
                deflate.Write(raw, 0, raw.Length);
            return output.ToArray();
        }
    }

    private static void WriteMaskMha(Dictionary<string, string> sourceMeta, byte[] compressedMask, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        var lines = new List<string>();
        foreach (string key in new string[] {
            "ObjectType", "NDims", "BinaryData", "BinaryDataByteOrderMSB",
            "TransformMatrix", "Offset", "CenterOfRotation", "AnatomicalOrientation",
            "ElementSpacing", "DimSize"
        })
            if (sourceMeta.ContainsKey(key)) lines.Add(key + " = " + sourceMeta[key]);
        if (!sourceMeta.ContainsKey("BinaryData")) lines.Add("BinaryData = True");
        if (!sourceMeta.ContainsKey("BinaryDataByteOrderMSB")) lines.Add("BinaryDataByteOrderMSB = False");
        lines.Add("CompressedData = True");
        lines.Add("CompressedDataSize = " + compressedMask.Length.ToString(CultureInfo.InvariantCulture));
        lines.Add("ElementType = MET_UCHAR");
        lines.Add("ElementDataFile = LOCAL");
        byte[] header = Encoding.ASCII.GetBytes(string.Join("\n", lines) + "\n");
        using (var fs = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            fs.Write(header, 0, header.Length);
            fs.Write(compressedMask, 0, compressedMask.Length);
        }
    }

    private static int[] ParseDims(string dimSize)
    {
        var parts = dimSize.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 3) throw new InvalidDataException("Expected 3D DimSize");
        return new int[] {
            int.Parse(parts[0], CultureInfo.InvariantCulture),
            int.Parse(parts[1], CultureInfo.InvariantCulture),
            int.Parse(parts[2], CultureInfo.InvariantCulture)
        };
    }

    private static string GetMeta(Dictionary<string, string> meta, string key)
    {
        return meta.ContainsKey(key) ? meta[key] : "";
    }

    private static string ToCsv(Row s)
    {
        return string.Join(",", new string[] {
            Csv(s.CaseId), s.BeamIdx.ToString(CultureInfo.InvariantCulture), s.CpIdx.ToString(CultureInfo.InvariantCulture),
            Csv(s.Definition), Csv(s.DosePath), Csv(s.MaskPath), Csv(s.DimSize), Csv(s.ElementSpacing),
            D(s.DoseMax), D(s.ThresholdValue), s.DilationVoxels.ToString(CultureInfo.InvariantCulture),
            s.Voxels.ToString(CultureInfo.InvariantCulture), s.PositiveVoxels.ToString(CultureInfo.InvariantCulture),
            D(s.PositivePercent), s.OutputBytes.ToString(CultureInfo.InvariantCulture), Csv(s.Status), Csv(s.Notes)
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
$result = [AuxMaskDefinitionEval]::Process(
    $training.Path,
    (Join-Path (Get-Location) $OutputDir),
    (Join-Path (Get-Location) $StatsCsv),
    $MaxFiles,
    $SampleMode,
    $MaxDegreeOfParallelism,
    [bool]$WriteMasks,
    [bool]$Force
)

[PSCustomObject]@{
    TrainingDir = $training.Path
    OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
    StatsCsv = (Resolve-Path -LiteralPath $StatsCsv).Path
    MaxFiles = $MaxFiles
    SampleMode = $SampleMode
    MaxDegreeOfParallelism = $MaxDegreeOfParallelism
    WriteMasks = [bool]$WriteMasks
    DoseFiles = $result.DoseFiles
    Rows = $result.Rows
    ProcessedRows = $result.ProcessedRows
    ErrorRows = $result.ErrorRows
    OutputGB = [Math]::Round($result.OutputBytes / 1GB, 3)
}
