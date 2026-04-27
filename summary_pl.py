"""Generate PDF summary of KroneckerMarkerCovariance method in Polish."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import HRFlowable


OUTPUT = "kronecker_marker_summary_pl.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
    topMargin=2.5 * cm,
    bottomMargin=2.5 * cm,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title",
    parent=styles["Normal"],
    fontSize=16,
    fontName="Helvetica-Bold",
    spaceAfter=6,
    alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    fontName="Helvetica",
    textColor=colors.HexColor("#555555"),
    spaceAfter=16,
    alignment=TA_CENTER,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Normal"],
    fontSize=12,
    fontName="Helvetica-Bold",
    spaceBefore=14,
    spaceAfter=4,
    textColor=colors.HexColor("#1a1a2e"),
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica",
    leading=15,
    spaceAfter=6,
)
eq_style = ParagraphStyle(
    "Eq",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Courier",
    leading=14,
    leftIndent=20,
    spaceBefore=4,
    spaceAfter=4,
    backColor=colors.HexColor("#f5f5f5"),
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica",
    leading=14,
    leftIndent=16,
    spaceAfter=3,
    bulletIndent=0,
)


def h2(text):
    return Paragraph(text, h2_style)


def body(text):
    return Paragraph(text, body_style)


def eq(text):
    return Paragraph(text, eq_style)


def bullet(text):
    return Paragraph(f"&bull;  {text}", bullet_style)


def gap(n=6):
    return Spacer(1, n)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)


story = [
    Paragraph("Kowariancja z Kroneckerem i markerami", title_style),
    Paragraph("Krótkie podsumowanie metody — Multiplex Image Model", subtitle_style),
    hr(),

    # --- Problem ---
    h2("1. Problem"),
    body(
        "Rekonstruujemy obraz multiplex złożony z <b>C markerów</b> i <b>H×W pikseli</b>. "
        "Model predykuje średnią μ i niepewność σ per piksel per marker. "
        "Celem jest modelowanie <b>korelacji przestrzennych i między markerami</b> — "
        "nie zakładamy niezależności pikseli."
    ),

    # --- Struktura kowariancji ---
    h2("2. Struktura kowariancji"),
    body(
        "Definiujemy kowariancję na przestrzeni <b>NC wymiarów</b> "
        "(N = H·W pikseli, C markerów):"
    ),
    gap(),
    eq("K = (Kx ⊗ Ky) ⊗ K_C  +  U_block · U_block\u1d40  +  \u03b5I"),
    gap(10),

    body("<b>Kx, Ky ∈ ℝ^{n×n}</b> — jądro Matérna 1D na osiach przestrzennych:"),
    bullet("Separowalna aproksymacja izotropowego Matérna."),
    bullet("Kx = Ky — ta sama siatka n punktów równomiernie w [0, 1]."),
    bullet("ν = 1.5 (raz różniczkowalne), lengthscale = 5.0 (szeroka korelacja)."),
    gap(4),

    body("<b>K_C ∈ ℝ^{C×C}</b> — kowariancja markerów:"),
    eq("K_C = E · E\u1d40  +  \u03b4I"),
    body(
        "gdzie E ∈ ℝ^{C×D} to znormalizowane rzutowanie embedingów Hyperkernel "
        "(nn.Linear → normalize po wierszach). "
        "Normalizacja wierszy sprawia, że K_C jest macierzą korelacji "
        "(jedynki na diagonali), co ogranicza liczbę uwarunkowania do max C."
    ),
    gap(4),

    body("<b>U_block ∈ ℝ^{NC×C}</b> — niskorangowy składnik per piksel (z dekodera)."),
    body("<b>εI</b> — jitter numeryczny dla stabilności."),

    # --- Obliczenia ---
    h2("3. Efektywne obliczenia"),
    body(
        "Macierz NC×NC <b>nie jest nigdy materializowana</b> "
        "(dla N = 112², C = 40 miałaby ~2×10\u2077 wymiarów). "
        "Korzystamy z rozkładów spektralnych:"
    ),
    gap(),
    eq("Kx = V \u039bx V\u1d40,    Ky = V \u039by V\u1d40,    K_C = Vc \u039bC Vc\u1d40"),
    gap(4),
    body("Wartości własne złożonego składnika Kroneckerowskiego:"),
    eq("\u03bbijk = \u03bbx_i · \u03bby_j · \u03bbC_k  +  \u03b5"),
    gap(4),
    body(
        "Rozwiązanie A⁻¹v (gdzie A = (Kx⊗Ky)⊗K_C + εI) sprowadza się do "
        "<b>sześciu operacji einsum</b>: transformacja do bazy eigenvektorów, "
        "skalowanie przez 1/λijk, transformacja z powrotem."
    ),
    gap(4),
    body("Człon niskorangowy obsługuje <b>tożsamość Woodbury</b>:"),
    eq("K\u207b\u00b9e = A\u207b\u00b9e \u2212 A\u207b\u00b9U(I + U\u1d40A\u207b\u00b9U)\u207b\u00b9U\u1d40A\u207b\u00b9e"),
    eq("log det K = log det A + log det(I + U\u1d40A\u207b\u00b9U)"),
    gap(4),
    body("<b>Złożoność:</b>"),
    bullet("O(n³) — raz przy inicjalizacji (rozkład Kx, Ky)."),
    bullet("O(C³) — per batch (rozkład K_C z embedingów markerów)."),
    bullet("O(n²·C) — per obraz (solver A⁻¹, Woodbury)."),

    # --- Stabilność ---
    h2("4. Stabilność numeryczna"),
    body("Dwa kluczowe zabiegi niezbędne do zbieżności:"),
    bullet(
        "<b>Normalizacja wierszy E</b>: bez niej K_C ma liczbę uwarunkowania ~10⁵–10⁶ "
        "→ GP NLL = nan od pierwszej epoki."
    ),
    bullet(
        "<b>float64 dla eigh(K_C)</b>: gdy C > D (wymiar projekcji), "
        "K_C ma C−D powtarzających się wartości własnych dokładnie równych δ. "
        "LAPACK w float32 nie zbiega — rzutowanie do float64 i z powrotem rozwiązuje problem."
    ),

    # --- Wyniki ---
    h2("5. Wyniki wstępne"),
    body(
        "Po 24 epokach (ImVs-19, batch_size=8, λ_GP=0.1, lengthscale=5.0):"
    ),
    bullet("MAE: 0.127 → 0.030 (postępująca poprawa rekonstrukcji)."),
    bullet(
        "Pearson ρ(MAE, Var) ≈ 0.90–0.95 — model dobrze kalibruje niepewność: "
        "wysoka predykowana wariancja koreluje z wysokim błędem rekonstrukcji."
    ),
    bullet("GP NLL stale ujemny i malejący — kowariancja markerów aktywnie się uczy."),
    bullet("Liczba uwarunkowania K_C: ~1000–1500, min eigval = 0.01 — stabilna."),
]

doc.build(story)
print(f"Saved: {OUTPUT}")
