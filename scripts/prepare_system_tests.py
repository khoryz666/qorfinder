#!/usr/bin/env python3
"""Build the folders the system tests index, T1 (data/t1-mixed) and T7
(data/t7-change), from CUAD_v1.zip and the hand-made files in data/fixtures/.

Usage: python3 scripts/prepare_system_tests.py

T1 holds 10 CUAD PDF, 9 DOCX written from CUAD TXT, 10 CUAD TXT and the files
QorFinder must skip with a reason. T7 is left in its "before" state; the
changes the test then makes by hand are in results/system-tests.txt. Both
folders are replaced on every run. Needs CUAD_v1.zip in data/cuad/ (see
prepare_cuad.py); never touches the network.
"""

import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from prepare_cuad import open_cuad_zip

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "data/fixtures"
T1 = REPO / "data/t1-mixed"
T7 = REPO / "data/t7-change"

PDF = "CUAD_v1/full_contract_pdf/"
TXT = "CUAD_v1/full_contract_txt/"

T1_PDF = [
    "Part_I/Co_Branding/2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.pdf",
    "Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/TubeMediaCorp_20060310_8-K_EX-10.1_513921_EX-10.1_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate Agreement.pdf",
    "Part_I/Affiliate_Agreements/UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2.pdf",
]
# Cybergy is here as TXT as well as PDF on purpose: its clauses should come
# back with the same text and score from both extractors.
T1_TXT = [
    "BLUEHILLSBANCORP,INC_05_20_2014-EX-1.1-AGENCY AGREEMENT.txt",
    "BellringBrandsInc_20190920_S-1_EX-10.12_11817081_EX-10.12_Manufacturing Agreement1.txt",
    "CARDAX,INC_08_19_2014-EX-10.1-COLLABORATION AGREEMENT.txt",
    "CORALGOLDRESOURCES,LTD_05_28_2020-EX-4.1-CONSULTING AGREEMENT.txt",
    "CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development Agreement.txt",
    "CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.txt",
    "FOUNDATIONMEDICINE,INC_02_02_2015-EX-10.2-Collaboration Agreement.txt",
    "HealthcentralCom_19991108_S-1A_EX-10.27_6623292_EX-10.27_Co-Branding Agreement.txt",
    "KIROMICBIOPHARMA,INC_04_08_2020-EX-10.28-JOINT VENTURE AGREEMENT.txt",
    "LegacyEducationAllianceInc_20200330_10-K_EX-10.18_12090678_EX-10.18_Development Agreement.txt",
]
# Written as DOCX, one paragraph per line of the CUAD TXT.
T1_DOCX = [
    "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt",
    "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt",
    "ACCURAYINC_09_01_2010-EX-10.31-DISTRIBUTOR AGREEMENT.txt",
    "ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT.txt",
    "ADAPTIMMUNETHERAPEUTICSPLC_04_06_2017-EX-10.11-STRATEGIC ALLIANCE AGREEMENT.txt",
    "ADIANUTRITION,INC_04_01_2005-EX-10.D2-RESELLER AGREEMENT.txt",
    "ADMA BioManufacturing, LLC -  Amendment #3 to Manufacturing Agreement .txt",
    "ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT.txt",
    "AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT.txt",
]
# A PDF cut off after its first 4000 bytes, so its cross-reference table is missing.
CORRUPT_PDF_FROM = T1_PDF[1]
CORRUPT_PDF_BYTES = 4000
# Just over the 50 MB limit (52,428,800 bytes): 972,316 lines of 56 bytes.
OVER_50MB_LINE = b"This line pads the file past the fifty megabyte limit.\r\n"
OVER_50MB_LINES = 972_316
# Made by hand once and committed: a PDF with an open password, a PDF whose
# only page is an image, and a spreadsheet (an unsupported type).
T1_FIXTURES = ["encrypted.pdf", "scanned.pdf", "fees.xlsx"]

# T7's three files before the change, under short names.
T7_BEFORE = {
    "A-collaboration.txt": "CARDAX,INC_08_19_2014-EX-10.1-COLLABORATION AGREEMENT.txt",
    "B-consulting.txt": "CORALGOLDRESOURCES,LTD_05_28_2020-EX-4.1-CONSULTING AGREEMENT.txt",
    "C-development.txt": "CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development Agreement.txt",
}

# The smallest DOCX docx-rs and Word open: the content types, the package
# relationship, the document and its (empty) relationships.
CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)
DOCUMENT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
)


def write_docx(text: str, dest: Path) -> None:
    """A DOCX with one paragraph per line of `text`. Entries carry a fixed
    date, so the same text always gives the same bytes."""
    paragraphs = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>'
        for line in text.splitlines()
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as docx:
        for name, data in [
            ("[Content_Types].xml", CONTENT_TYPES),
            ("_rels/.rels", RELS),
            ("word/document.xml", document),
            ("word/_rels/document.xml.rels", DOCUMENT_RELS),
        ]:
            docx.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)


def fresh(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def build_t1(cuad: zipfile.ZipFile) -> None:
    fresh(T1)
    for path in T1_PDF:
        (T1 / Path(path).name).write_bytes(cuad.read(PDF + path))
    for name in T1_TXT:
        (T1 / name).write_bytes(cuad.read(TXT + name))
    for name in T1_DOCX:
        text = cuad.read(TXT + name).decode("utf-8")
        write_docx(text, T1 / (name.removesuffix(".txt").rstrip() + ".docx"))
    corrupt = cuad.read(PDF + CORRUPT_PDF_FROM)[:CORRUPT_PDF_BYTES]
    (T1 / "corrupt.pdf").write_bytes(corrupt)
    with (T1 / "over-50mb.txt").open("wb") as f:
        f.write(OVER_50MB_LINE * OVER_50MB_LINES)
    for name in T1_FIXTURES:
        shutil.copyfile(FIXTURES / name, T1 / name)
    print(f"wrote {len(list(T1.iterdir()))} files to {T1}")


def build_t7(cuad: zipfile.ZipFile) -> None:
    fresh(T7)
    for name, source in T7_BEFORE.items():
        (T7 / name).write_bytes(cuad.read(TXT + source))
    print(f"wrote {len(T7_BEFORE)} files to {T7} (the state before the change)")


def main() -> int:
    cuad = open_cuad_zip()
    build_t1(cuad)
    build_t7(cuad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
