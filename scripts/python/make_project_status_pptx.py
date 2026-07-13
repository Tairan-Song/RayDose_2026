"""Generate a small DoseRAD baseline update PPTX without external dependencies."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5


def emu(value_inch: float) -> int:
    return int(value_inch * EMU_PER_INCH)


def tx_box(
    shape_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    font_size: int = 22,
    bold: bool = False,
    color: str = "1F2937",
    fill: str | None = None,
    line: str | None = None,
) -> str:
    paragraphs = text.split("\n")
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    para_xml = []
    for paragraph in paragraphs:
        if paragraph.strip() == "":
            para_xml.append("<a:p/>")
            continue
        b_attr = ' b="1"' if bold else ""
        para_xml.append(
            "<a:p>"
            f'<a:r><a:rPr lang="en-US" sz="{font_size * 100}"{b_attr}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            "</a:rPr>"
            f"<a:t>{escape(paragraph)}</a:t></a:r>"
            "</a:p>"
        )
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          {fill_xml}
          {line_xml}
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" inset="91440" anchor="mid"/>
          <a:lstStyle/>
          {''.join(para_xml)}
        </p:txBody>
      </p:sp>
    """


def rect(
    shape_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fill: str,
    line: str = "CBD5E1",
    font_size: int = 18,
    color: str = "111827",
) -> str:
    return tx_box(shape_id, x, y, w, h, text, font_size=font_size, color=color, fill=fill, line=line)


def title(title_text: str, subtitle: str = "") -> list[str]:
    parts = [
        tx_box(100, 0.55, 0.35, 12.2, 0.55, title_text, font_size=28, bold=True, color="0F172A"),
        tx_box(101, 0.58, 0.95, 12.1, 0.3, subtitle, font_size=12, color="64748B"),
    ]
    parts.append(rect(102, 0.55, 1.32, 12.2, 0.03, "", fill="2563EB", line="2563EB"))
    return parts


def bullets(shape_id: int, x: float, y: float, w: float, h: float, items: list[str], font_size: int = 18) -> str:
    text = "\n".join(f"• {item}" for item in items)
    return tx_box(shape_id, x, y, w, h, text, font_size=font_size, color="1F2937")


def slide_xml(shapes: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def content_types(num_slides: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, num_slides + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>
"""


def package_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""


def presentation_xml(num_slides: int) -> str:
    sld_ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, num_slides + 1))
    master_rid = f"rId{num_slides + 1}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="{master_rid}"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="wide"/>
  <p:notesSz cx="{emu(10)}" cy="{emu(7.5)}"/>
</p:presentation>
"""


def presentation_rels(num_slides: int) -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, num_slides + 1)
    )
    rels += (
        f'\n  <Relationship Id="rId{num_slides + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="slideMasters/slideMaster1.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels}
</Relationships>
"""


def slide_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
"""


def slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def slide_layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def slide_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle/><p:bodyStyle/><p:otherStyle/>
  </p:txStyles>
</p:sldMaster>
"""


def slide_master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DoseRAD">
  <a:themeElements>
    <a:clrScheme name="DoseRAD">
      <a:dk1><a:srgbClr val="111827"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="334155"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1>
      <a:accent2><a:srgbClr val="22C55E"/></a:accent2>
      <a:accent3><a:srgbClr val="F59E0B"/></a:accent3>
      <a:accent4><a:srgbClr val="EF4444"/></a:accent4>
      <a:accent5><a:srgbClr val="8B5CF6"/></a:accent5>
      <a:accent6><a:srgbClr val="06B6D4"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="DoseRAD">
      <a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont>
      <a:minorFont><a:latin typeface="Aptos"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="DoseRAD">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def build_slides() -> list[str]:
    today = date.today().strftime("%B %d, %Y")
    slides: list[list[str]] = []

    slides.append(
        title("DoseRAD2026 Photon Task 1: Baseline Update", f"Local project status for collaborator discussion | {today}")
        + [
            tx_box(10, 0.8, 1.95, 11.7, 0.8, "Goal: predict per-beam/control-point 3D dose from CT and beam geometry.", 26, True, "111827"),
            rect(11, 1.0, 3.0, 3.1, 1.0, "Input\nCT + Bx/CPy parameters", "DBEAFE", "60A5FA", 18),
            rect(12, 5.1, 3.0, 3.1, 1.0, "Model\nGeometry-conditioned 3D U-Net", "DCFCE7", "22C55E", 18),
            rect(13, 9.2, 3.0, 3.1, 1.0, "Output\nDose_Bx_CPy.mha", "FEF3C7", "F59E0B", 18),
            tx_box(14, 1.0, 4.55, 11.4, 1.2, "Current status: baseline pipeline is complete and has produced first local validation results. Performance is still weak; the next phase is model improvement.", 20, False, "334155", "F8FAFC", "CBD5E1"),
        ]
    )

    slides.append(
        title("Data And Preprocessing", "What is already prepared locally")
        + [
            bullets(
                20,
                0.8,
                1.75,
                5.7,
                4.5,
                [
                    "Photon training dataset is downloaded locally.",
                    "Case-level split is complete: 60 train cases / 15 validation cases.",
                    "Preprocessing supports CT HU normalization.",
                    "Optional HU-to-density conversion is implemented.",
                    "Dose masks and dose statistics are available for support-region evaluation.",
                ],
                18,
            ),
            rect(21, 7.0, 1.75, 4.8, 0.75, "Data split", "E0F2FE", "0284C7", 20),
            rect(22, 7.0, 2.75, 4.8, 0.75, "CT / density preprocessing", "E0F2FE", "0284C7", 20),
            rect(23, 7.0, 3.75, 4.8, 0.75, "64 x 64 x 64 training crop", "E0F2FE", "0284C7", 20),
            rect(24, 7.0, 4.75, 4.8, 0.75, "Dose and mask loading", "E0F2FE", "0284C7", 20),
            tx_box(25, 7.0, 5.85, 4.8, 0.55, "Split file: splits/photon_case_split.csv", 15, False, "475569"),
        ]
    )

    slides.append(
        title("Baseline Model", "Geometry-conditioned 3D U-Net")
        + [
            rect(30, 0.9, 1.7, 3.0, 0.9, "CT volume\n1-channel 3D image", "DBEAFE", "3B82F6", 18),
            rect(31, 0.9, 3.0, 3.0, 1.2, "Condition vector\nbeam, CP, gantry,\nisocenter, MLC", "EDE9FE", "8B5CF6", 18),
            rect(32, 4.9, 2.25, 3.5, 1.3, "FiLM-conditioned\n3D U-Net", "DCFCE7", "22C55E", 22),
            rect(33, 9.4, 2.25, 3.0, 1.3, "Predicted dose\nDose_Bx_CPy.mha", "FEF3C7", "F59E0B", 20),
            bullets(
                34,
                1.0,
                4.95,
                11.5,
                1.3,
                [
                    "Energy spectrum is optional for ablation, but current main baseline is no-energy.",
                    "Model output was changed from ReLU to linear; exported dose is clamped to non-negative.",
                ],
                17,
            ),
        ]
    )

    slides.append(
        title("Current Baseline Experiment", "First reportable local run")
        + [
            rect(40, 0.8, 1.65, 3.9, 0.65, "Experiment", "F8FAFC", "94A3B8", 18),
            tx_box(41, 1.0, 2.45, 5.1, 3.4, "Name: baseline_hu_no_energy_v2_linear\nInput: CT HU, no energy spectrum\nCrop: 64 x 64 x 64\nTrain samples: 1000\nValidation samples: 128\nEpochs: 5\nExported predictions: 64", 18),
            rect(42, 7.0, 1.65, 4.8, 0.65, "Generated artifacts", "F8FAFC", "94A3B8", 18),
            bullets(
                43,
                7.0,
                2.45,
                5.2,
                3.4,
                [
                    "best.pt checkpoint",
                    "Dose_Bx_CPy.mha predictions",
                    "prediction_manifest.csv",
                    "per-sample metrics CSV",
                    "summary metrics CSV",
                ],
                18,
            ),
        ]
    )

    slides.append(
        title("Performance Summary", "Full-volume exported validation predictions, n = 64")
        + [
            rect(50, 0.8, 1.65, 3.6, 0.8, "masked relative MAE\n13.12%", "FEE2E2", "EF4444", 20),
            rect(51, 4.85, 1.65, 3.6, 0.8, "gamma 3%/3mm\n~0.0000007", "FEE2E2", "EF4444", 20),
            rect(52, 8.9, 1.65, 3.6, 0.8, "prediction time\n0.112 s/sample", "DCFCE7", "22C55E", 20),
            bullets(
                53,
                0.9,
                3.1,
                11.7,
                2.3,
                [
                    "The pipeline works end-to-end, but model quality is still weak.",
                    "Gamma pass rate is essentially zero, so spatial dose agreement is not clinically useful yet.",
                    "The model still underfits and tends to predict low-amplitude dose.",
                    "Full-image MAE is misleading because most voxels are near zero; masked/high-dose metrics matter more.",
                ],
                18,
            ),
            tx_box(54, 0.9, 5.9, 11.7, 0.45, "Key file: outputs/baseline_hu_no_energy_v2_linear/evaluate_exported/exported_prediction_summary.csv", 14, False, "475569"),
        ]
    )

    slides.append(
        title("Next Steps", "Recommended discussion points")
        + [
            bullets(
                60,
                0.9,
                1.65,
                5.7,
                4.8,
                [
                    "Increase mask_weight to emphasize the dose-support region.",
                    "Add a beam geometry mask/channel as an explicit spatial prior.",
                    "Compare HU vs HU-to-density input.",
                    "Compare no-energy vs with-energy conditioning.",
                    "Try sliding full-volume inference instead of crop insertion.",
                    "Train longer with more samples after the above changes.",
                ],
                18,
            ),
            rect(61, 7.2, 1.8, 4.8, 0.8, "Immediate priority", "DBEAFE", "3B82F6", 20),
            tx_box(62, 7.35, 2.85, 4.5, 1.25, "Improve dose-support learning before scaling up expensive training.", 22, True, "111827"),
            tx_box(63, 7.35, 4.35, 4.5, 1.4, "Success signal: gamma pass rate and masked relative MAE improve, not only full-volume MAE.", 18, False, "334155", "F8FAFC", "CBD5E1"),
        ]
    )

    return [slide_xml(slide) for slide in slides]


def write_pptx(path: Path) -> None:
    slides = build_slides()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", package_rels())
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide)
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels())


def main() -> None:
    output = Path("outputs/presentations/DoseRAD_Baseline_Update_20260710.pptx")
    write_pptx(output)
    print(output)


if __name__ == "__main__":
    main()
