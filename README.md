# Project of Croatian Sentiment Reviews 

Članovi skupine:


* Katja Bešlić

* Viktorija Borko

* Ana Domović

* Goran Mihalković

* Lucija Poslek

Opis:
--
Projekt je rad studenata kolegija Obrada prirodnog jezika studija informacijskih znanosti. Cilj je izrada korpusa na hrvatskom jeziku, analiza i klasifikacija sentimenata. Temelji se na vlastitim prikupljenim podacima i klasificiji rečenica te obradom i izračunom podataka koristeći modele strojnog i dubokog učenja te transformatora. 

Korpus:
--
* Jezik: hrvatski
* Domena: recenzije doktora (dentalne) medicine - zdravstvo 
* Veličina korpusa: 4070

Za prikupljanje sadržaja korpusa korištena je web-stranica [najdoktor.com](najdoktor.com). Portal omogućava korisnicima komentiranje rada i stručnosti liječnika privatnog i javnog sektora iz različitih medicinskih domena. Komentari služe kao recenzije koje se potom označuju nekom od oznaka, poput pozitivno, negativno i mješovito. Komentari su javno dostupni, a podaci komentatora, poput korisničkog imena, se ne prikupljaju.
Korpus sadržava 4070 rečenica uzetih iz 825 komentara o 25 liječnika.

Pilot anotiranje:
--

Projekt je zahtijevao provođenje _pilot_-anotacijske kampanje prije konačnog označavanja cijelog skupa. Za potrebe ovog pilota odabran je reprezentativni nasumiačan uzorak od 150 rečenica iz našeg korpusa. Zadatak svakog člana tima bio je neovisno analizirati i označiti emocionalni ton (sentiment) svake rečenice. U našem specifičnom projektu, anotacijska shema proširena je na pet kategorija, čime smo obuhvatili sljedeće tonove: pozitivan, negativan, sarkastičan, mješoviti (mixed) i neutralan. Četiri člana su provela osnovnu anotaciju, dok je peti član čiji je zadatak bio donijeti konačnu odluku u visoko ambivalentnim situacijama nakon konzultacije s grupom.

Anotiranje i eksploratorna analiza:
--

Nakon uspješno završenog _pilot_ anotiranja i usvajanja konačnih smjernica, pokrenuta je finalna kampanja kompletne anotacije cjelokupnog korpusa od 4070 rečenica. Kako bi se osigurala objektivnost, odredili smo četiri anotatora po rečenici, a peti je član tima preuzeo ulogu glavnog anotatora podataka (data curator) čime je svakoj rečenici dodijeljena konačna oznaka. Konačna oznaka za svaku rečenicu izvedena je principom većinskog glasovanja. Kao izravni rezultati ove faze generirane su dvije ključne stvari: četiri stupca datoteke sadrži podatkovni skup sa svim pojedinačnim ocjenama i tragovima svih anotatora te peti stupac koji predstavlja pročišćeni skup s jednom, konačno usvojenom i agregiranom oznakom tona po rečenici. Anotirani korpus dostupan je kao .csv datoteka u mapi "korpus datoteke" u cjelovitom, pojedinačnom i grupiranom izdanju.

Fiksiranjem konačnih oznaka započeli smo s provedbom detaljne eksploratorne analize podataka (EDA). Izračunata je distribucija klasa (točan broj pozitivnih, negativnih, neutralnih, sarkastičnih i mješovitih rečenica), kao i statistika duljine rečenica (prosječan broj riječi, te identifikacija najkraće i najduže rečenice u cijelom korpusu). Podaci su potom, korištenjem biblioteke sklearn, podijeljeni na skupove za treniranje (80%), validaciju (5% do 10%) i testiranje (10% do 15%) , a dobivene podjele stavljene su na raspolaganje ostalim grupama i pohranjene u mapu za zajedničko korištenje. Statistika je dostpuna u mapi 


Treniranje:
--
Završna i tehnički najzahtjevnija faza projekta obuhvaćala je implementaciju i testiranje triju različitih pristupa strojnom učenju i obradi prirodnog jezika kako bi se automatizirala klasifikacija tona recenzija. Dva člana bila su zadužena za klasične strojne modele (ML), dva za duboko učenje (DL), dok je posljednji član radio na finom podešavanju transformatorskih modela. Svi modeli trenirani su kako na našem specifičnom skupu podataka, tako i na kombiniranom zajedničkom skupu svih grupa. Evaluacija je izvršena na tri neovisna testna skupa radi provjere robusnosti i generalizacije modela. Za svaki model precizno su izračunate metrike: točnost (accuracy), preciznost (precision), odziv (recall) i ključna F1-mjera, a izgrađene su i matrice zabune (confusion matrix) za vizualizaciju pogrešaka.

### Metode:

Strojno učenje (ML): 
* SVM
* Logistic Regression

Plitko duboko učenje (SDL):
* GRM
* LSTM

Transformeri:
* BERTić
* Gemma 2
* CroSlo




