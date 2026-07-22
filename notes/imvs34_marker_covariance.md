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
