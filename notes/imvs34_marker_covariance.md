# ImVs-34 (kronecker-learnmask) — analiza kowariancji markerów K_C

Notatka / wiadomość do grupy. Panel hn (40 markerów); danenberg i hoch-rna analogicznie.
Skrypt: `dump_marker_covariance.py --panel <dataset>`. Figury w `logs/marker_covariance_ImVs-34_*.png`.

---

Cześć,

Podzielę się wynikiem małej analizy modelu ImVs-34 (wariant kronecker-learnmask). Chciałem sprawdzić, czy ten model nauczył się jakiejś ciekawej reprezentacji niepewności, której nie ma zwykły GP. Kalibracja marginalna (Pearson log-MSE vs log-sigma oraz log-sigma vs log-MAE) wychodziła praktycznie identyczna jak w bazowym GP, więc zajrzałem bezpośrednio w to, co ten wariant realnie dodaje.

Krótko o co chodzi, dla tych co nie siedzieli w części GP: w losie GP, oprócz kowariancji przestrzennej po pikselach, jest dodatkowo kowariancja po markerach — macierz K_C. Powstaje ona z embeddingów markerów z hyperkernela (rzutowanych liniowo i znormalizowanych): K_C = E·Eᵀ. Bazowy model traktuje markery jako niezależne, czyli de facto K_C = I. Ważne: K_C nie zależy od obrazka — to czysty lookup po embeddingach markerów, więc można ją policzyć raz i po prostu obejrzeć.

W załączniku K_C dla panelu hn (40 markerów), dwa panele:
- lewy: pełna K_C,
- prawy: K_C po odjęciu dominującej wspólnej składowej („residual").

Kolor = korelacja między dwoma markerami w przestrzeni embeddingów (czerwony dodatni, niebieski ujemny). Markery na obu osiach są ułożone tak samo — uporządkowane przez klasteryzację panelu residualnego, żeby podobne markery leżały obok siebie i tworzyły bloki (to tylko kolejność osi, nie zmienia wartości).

Co widać:
- Lewy panel jest głównie czerwony — K_C jest zdominowana przez jedną wspólną składową (pierwszy wektor własny to ~55% macierzy). Wszystko koreluje dodatnio, co jest bardzo blisko tego, co dostalibyśmy w ogóle bez kowariancji markerów.
- Dopiero prawy panel (po odjęciu tej składowej) pokazuje właściwą strukturę: duży blok limfoidalny (FOXP3, PD1, CD27, ICOS, LAG3, CD20 — wszystkie mocno dodatnio) przeciwstawiony blokowi mieloidalnemu (CD11c, CD16, MPO; plus cl.PARP), który jest niebieski względem limfoidalnego. Czyli model ustawił limfoidalne vs mieloidalne na jednej osi — sensowna biologicznie struktura, której bazowy model (K_C = I) w ogóle nie jest w stanie wyrazić.
- Sanity check: DNA1 i DNA2 lądują jako osobna para (~+1), czyli dwa barwienia jądrowe rozpoznane jako praktycznie identyczne. Markery housekeeping/jądrowe (Histone H3, Ki67, SMA, B2M) są blade, niezależne od osi immunologicznej — co też ma sens.

Sprawdziłem też, skąd ta struktura pochodzi, i to jest ciekawe: **nie jest odziedziczona z rekonstrukcji**. Surowe embeddingi hyperkernela (te same, które ma model bazowy) są nieustrukturyzowane — niemal pełnorzędowe i wzajemnie prawie ortogonalne, korelacje ~±0.03, żadnych bloków (rekonstrukcja pcha embeddingi ku odrębnym filtrom per marker, a nie ku grupowaniu). Cała struktura limfoidalna/mieloidalna powstaje dopiero w warstwie projekcji (`embedding_projection`), która (a) w modelu bazowym w ogóle nie istnieje i (b) jest trenowana wyłącznie przez loss K_C. Korelacja między strukturą surowych a rzutowanych embeddingów to ~0.08, czyli praktycznie zero. Innymi słowy: to grupowanie jest realną „zasługą" kowariancji markerów, a nie efektem ubocznym rekonstrukcji.

Danenberg i hoch-rna wyglądają analogicznie: danenberg dokłada parę stromalną FSP1–Podoplanin, a hoch-rna to panel RNA, więc rzadkie chemokiny + kontrola DapB zlewają się w jedną grupę.

Wniosek: K_C faktycznie nauczyła się nietrywialnej, biologicznie sensownej struktury po markerach — i to struktury specyficznej dla mechanizmu kowariancji markerów, nie czegoś, co model bazowy też by miał. ALE dwie rzeczy tłumaczą, czemu nie widać tego w naszej kalibracji:
1. Ta struktura jest drugorzędna — dominuje wspólny „globalny" komponent, który działa niemal jak brak kowariancji markerów.
2. Co ważniejsze: K_C wchodzi tylko do losa (łączny log-likelihood po pikselach × markerach), a niepewność, którą raportujemy i kalibrujemy, to marginalna wariancja z głowicy dekodera (logvar), a nie wariancja a posteriori GP. Czyli nasze wykresy kalibracyjne strukturalnie nie mogą tego „zobaczyć" — dlatego wychodzą identyczne jak baseline.

Gdybyśmy chcieli pokazać efekt K_C na samej niepewności, trzeba by albo policzyć wariancję predykcyjną GP (która realnie używa K_C), albo zrobić test łączny — np. czy markery, które K_C grupuje razem, mają skorelowane błędy w leave-one-out.

Skrypt (dump_marker_covariance.py) jest w repo, liczy się na CPU w kilka sekund, przyjmuje --panel <dataset>. Dajcie znać co myślicie.

---

_Uwaga do potwierdzenia: fragment o pochodzeniu struktury opiera się na dowodzie pośrednim (surowe embeddingi samego ImVs-34). Twarde potwierdzenie = ta sama analiza na checkpoincie modelu bazowego (GP bez marker covariance)._

---

## Test: czy K_C przewiduje skorelowane błędy LOO?

Skrypt `test_kc_error_corr.py` (149 rekonstrukcji LOO, panel hn, 40 markerów). Liczy
empiryczną korelację map residuów (recon − target) między markerami, uśrednioną po
obrazach, i porównuje ją z K_C (pełnym i residualnym), z testem permutacyjnym.

**Wynik: NIE — K_C nie przewiduje skorelowanych błędów LOO.**

```
corr(K_C pełne,    korelacja-błędów pełna)     = -0.016   (praktycznie zero)
corr(K_C residual, korelacja-błędów residual)  = -0.194   (perm p = 0.025; null |r| max 0.083)
```

Pełne K_C: brak związku. Residualne K_C: słaby, istotny, ale UJEMNY — markery grupowane
przez K_C mają odrobinę *mniej* skorelowane błędy, odwrotnie niż hipoteza „K_C łapie
kowariancję błędów". Parami:

| para | K_C_resid | błąd_resid |
|---|---|---|
| DNA1 – DNA2 | +0.999 | −0.118 |
| CD163 – CD206 | +0.351 | −0.080 |
| CD163 – CD68 | +0.273 | −0.073 |
| CD14 – CD163 | +0.176 | −0.162 |

**Mechanizm (DNA1/DNA2):** K_C = +0.999, bo to prawie identyczne barwienia. Ale w LOO,
maskując DNA1, model ma DNA2 na wejściu → odtwarza DNA1 kopiując DNA2 → mały błąd (i
odwrotnie). Czyli K_C mierzy **podobieństwo/redundancję** markerów, a nie kowariancję
błędów; redundantne markery łatwo zaimputować z siebie → mały, zdekorelowany błąd. K_C
wiąże się więc raczej z **wielkością** błędu (grupa → niski błąd) niż z jego korelacją.

**Wniosek:** K_C to ciekawa wyuczona reprezentacja relacji markerów (biologicznie sensowna,
CD45RA/CD45RO ≈ 0.30), ale **nie działa jako operacyjny predyktor łącznego zachowania
błędów / skorelowanej niepewności**. „Ciekawa reprezentacja" ≠ „ciekawa niepewność"
w sensie mierzalnym na wyjściu — spójne z tym, że raportowana niepewność nie płynie z K_C.

Naturalny następny test: czy markery z wieloma sąsiadami w K_C rekonstruują się lepiej
w LOO (redundancja → niższy MSE) — to wielkość, z którą K_C faktycznie się wiąże.

---

## Test: czy redundancja w K_C przewiduje niższe MSE w LOO?

Skrypt `test_kc_redundancy_mse.py`. Per marker liczy „redundancję" z K_C i koreluje ją ze
średnim MSE per marker z CSV LOO (149 obrazów, 40 markerów). Jeden punkt korelacji = jeden
marker (n = 40). Pearson (recon–target, niezależny od skali) dodany jako kontrola confoundu.

**Wynik: TAK na surowym MSE — ale ⚠ patrz sekcja niżej: na metryce niezależnej od skali
efekt znika (był confoundem skali).** Markery z bliskim „bliźniakiem" w K_C mają niższe surowe MSE.

```
score                             vs MSE:  Spearman   Pearson
kc_max  (najlepszy bliźniak)               -0.878     -0.771    <- najsilniejszy
kc_nn05 (# sąsiadów > 0.5)                 -0.684     -0.535
kc_mean (średnie podobieństwo)             -0.480     -0.245
```

Najsilniejszy predyktor to `kc_max` — do imputacji zamaskowanego markera wystarczy jeden
bardzo podobny marker na wejściu, z którego można „skopiować".

Ranking:
- redundantne (kc_max ≈ 0.99): CD4, ICOS, PD1, CD27, FOXP3, CD3, LAG3, CD20 → MSE 0.0006–0.0036
- unikalne (kc_max ≈ 0, 0 sąsiadów): Ki67 (0.032), CD15 (0.016), Ecad (0.019) → najtrudniejsze (wyjątek: SMA)
- niuans: MPO/cl.PARP/CD16/CD11c mają ujemne kc_mean (anty-limfoidalne), ale kc_max ≈ 0.98
  (bliźniacy we własnym klastrze mieloidalnym) → niskie MSE. Liczy się posiadanie *jakiegokolwiek*
  bliźniaka (kc_max), nie ogólne podobieństwo do panelu (kc_mean).

**Zastrzeżenie (confound intensywności):** związek silny z MSE, ale słaby z Pearsonem
(kc_max vs pearson: Spearman −0.12). MSE zależy od skali, więc część efektu to fakt, że
redundantne markery immunologiczne bywają niżej-sygnałowe. Ale mechanizm bliźniaka jest realny
i niezależny od skali (DNA1/DNA2 i klaster mieloidalny są jasne, a mimo to mają niskie MSE).

## Test: kontrola confoundu — metryka niezależna od skali

Skrypt `test_kc_redundancy_nmse.py`. Zamiast surowego MSE używa NMSE = MSE/Var(target)
(= 1 − R²) oraz Pearsona(recon, target), liczonych z NPZ — obie niezależne od dynamiki markera.

**Wynik: efekt redundancji ZNIKA.** Poprzednie −0.88 (kc_max vs MSE) było prawie w całości
confoundem skali.

```
score     vs        Spearman   (chcemy)
kc_max    NMSE        +0.008       -      -> zero
kc_max    R^2         -0.008       +      -> zero
kc_max    pearson     -0.153       +      -> słabo, zły kierunek
kc_nn05   pearson     -0.309       +      -> słabo, zły kierunek
```

| marker | kc_max | R² | pearson |
|---|---|---|---|
| DNA1 / DNA2 | 0.999 | 0.97 | 0.99 |
| PD1 | 1.000 | 0.11 | 0.50 |
| LAG3 | 0.999 | 0.08 | 0.37 |
| cl.PARP | 0.999 | 0.01 | 0.21 |
| CD14 (unikalny) | 0.214 | 0.60 | 0.83 |
| HLADR (unikalny) | 0.136 | 0.54 | 0.81 |
| Ecad (unikalny) | 0.097 | 0.49 | 0.79 |

- Tylko prawdziwe duplikaty działają: DNA1/DNA2 (kc_max ≈ 1 **i** R² ≈ 0.97).
- Reszta klastra limfoidalnego (PD1, LAG3, cl.PARP): kc_max ≈ 1, ale R² 0.01–0.11 — podobieństwo
  embeddingów ≠ kopiowalność pikseli.
- Unikalne markery (Ecad, HLADR, CD14) bywają lepiej odtwarzalne niż redundantne limfoidalne.

Redundantne w K_C to po prostu rzadkie, niskosygnałowe markery immunologiczne → małe MSE
mechanicznie (mała wariancja), a nie „łatwe do zaimputowania".

## Spięcie wszystkich testów — operacyjne znaczenie K_C (wersja finalna)

1. K_C uczy się sensownej biologicznie **struktury podobieństwa markerów** (limfoidalne/mieloidalne;
   CD45RA/CD45RO ≈ 0.30, zgodne z paperem ImmuVis). ✓
2. **Nie** przewiduje skorelowanych błędów LOO (r ≈ 0 / słabo ujemne). ✗
3. **Nie** przewiduje jakości rekonstrukcji na metryce niezależnej od skali (NMSE/R²/Pearson ≈ 0);
   pozorny efekt na surowym MSE był confoundem skali. ✗ (teza o „mapie imputowalności" — WYCOFANA)

**Wniosek finalny:** K_C to interpretowalna wyuczona mapa podobieństwa markerów, ale **bez
wykrywalnego operacyjnego śladu na wyjściu modelu** — ani skorelowanej niepewności, ani realnej
jakości rekonstrukcji. Ciekawa reprezentacja, która (na razie) nie przekłada się na mierzalny
efekt w predykcjach. Jedyny czysty przypadek „podobieństwo → kopiowalność" to dosłowne
duplikaty (DNA1/DNA2).
