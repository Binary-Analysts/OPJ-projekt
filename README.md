# Project of Croatian Sentiment Reviews 

Članovi skupine:


* Katja Bešlić

* Viktorija Borko

* Ana Domović

* Goran Mihalković

* Lucija Poslek

Opis
--
Projekt je rad studenata kolegija Obrada prirodnog jezika studija informacijskih znanosti. Cilj je izrada korpusa na hrvatskom jeziku, analiza i klasifikacija sentimenata. Temelji se na vlastitim prikupljenim podacima i klasificiji rečenica te obradom i izračunom podataka koristeći modele strojnog i dubokog učenja te transformatora. 

Korpus
--
* Jezik: hrvatski
* Domena: zdravstvo - recenzije doktora medicine
* Veličina korpusa: 4070

Za prikupljanje sadržaja korpusa korištena je web-stranica [najdoktor.com](najdoktor.com). Portal omogućava korisnicima komentiranje rada i stručnosti liječnika privatnog i javnog sektora iz različitih medicinskih domena. Komentari služe kao recenzije koje se potom označuju nekom od oznaka, poput pozitivno, negativno i mješovito. Komentari su javno dostupni, a podaci komentatora, poput korisničkog imena, se ne prikupljaju.
Korpus sadržava 4070 rečenica uzetih iz 825 komentara o 25 liječnika.

_Pilot_-anotiranje
--

Projekt je zahtijevao provođenje _pilot_-anotacijske kampanje prije konačnog označavanja cijelog skupa. Za potrebe ovog pilota odabran je reprezentativni nasumičan uzorak od 150 rečenica iz našeg korpusa. Zadatak svakog člana tima bio je neovisno analizirati i označiti emocionalni ton (sentiment) svake rečenice. U našem specifičnom projektu, anotacijska shema proširena je na pet kategorija, čime smo obuhvatili sljedeće tonove: pozitivan, negativan, sarkastičan, mješoviti (mixed) i neutralan. Četiri člana su provela osnovnu anotaciju, dok je peti član donio konačnu odluku u visoko ambivalentnim situacijama nakon konzultacije s grupom. Po završetku _pilot_-anotacije, izračunata je Fleissova Kappa kao mjera međuanotatorskog slaganja - 78%.

Anotiranje i eksploratorna analiza
--

Nakon uspješno završenog _pilot_-anotiranja i usvajanja konačnih smjernica, pokrenuta je finalna kampanja kompletne anotacije cjelokupnog korpusa od 4070 rečenica. Kako bi se osigurala objektivnost, odredili smo četiri anotatora po rečenici, a peti je član tima preuzeo ulogu glavnog anotatora podataka (data curator) čime je svakoj rečenici dodijeljena konačna oznaka. Konačna oznaka za svaku rečenicu izvedena je principom većinskog glasovanja. Kao izravni rezultati ove faze generirane su dvije ključne stvari: četiri stupca datoteke sadrži podatkovni skup sa svim pojedinačnim ocjenama i tragovima svih anotatora te peti stupac koji predstavlja pročišćeni skup s jednom, konačno usvojenom i agregiranom oznakom tona po rečenici. Za cjelokupni korpus također je izračunata Fleissova kappa koja iznosi 83%. Anotirani korpus dostupan je kao _.csv_ datoteka u mapi "korpus datoteke" u cjelovitom, pojedinačnom i grupiranom izdanju. 

U tablici je naveden broj rečenica prema sentimentu, a koje je zaključno odredio anotator podataka.

 | Pozitivne | Negativne | Neutralne | Mješovite | Sarkastične |
 | --------- | --------- | --------- | --------- | ----------- |
 |    2003   |    1198   |    665    |    134    |      62     |



Fiksiranjem konačnih oznaka započeli smo s provedbom detaljne eksploratorne analize podataka (EDA). Izračunata je distribucija klasa (točan broj pozitivnih, negativnih, neutralnih, sarkastičnih i mješovitih rečenica), kao i statistika duljine rečenica (prosječan broj riječi, te identifikacija najkraće i najduže rečenice u cijelom korpusu). Podaci su potom, korištenjem biblioteke _sklearn_, podijeljeni na skupove za treniranje (80%), validaciju (5% do 10%) i testiranje (10% do 15%), a dobivene podjele stavljene su na raspolaganje ostalim grupama i pohranjene u mapu za zajedničko korištenje. 


Treniranje
--
Završna i tehnički najzahtjevnija faza projekta obuhvaćala je implementaciju i testiranje triju različitih pristupa strojnom učenju i obradi prirodnog jezika kako bi se automatizirala klasifikacija tona recenzija. Dva člana bila su zadužena za klasične strojne modele (ML), dva za duboko učenje (DL), dok je posljednji član radio na finom podešavanju transformatorskih modela. Svi modeli trenirani su kako na našem specifičnom skupu podataka, tako i na kombiniranom zajedničkom skupu svih grupa. Evaluacija je izvršena na tri neovisna testna skupa radi provjere robusnosti i generalizacije modela. Za svaki model precizno su izračunate metrike: točnost (_accuracy_), preciznost (_precision_), odziv (_recall_) i ključna F1-mjera, a izgrađene su i matrice zabune (_confusion matrix_) za vizualizaciju pogrešaka.

### Metode

Odlučili smo se za rad s pet klasa koristeći tri metode - strojno učenje, duboko učenje i transformatore. Svi modeli dostupni su u istoimenim mapama, npr. "duboko učenje" sadržava modele GRM i LSTM.

Strojno učenje (ML): 
* SVM
* Logistic Regression

Napravili smo 4 modela (1 za LR – pojedinačni train setovi testirani na pojedinačnim test setovima (svake grupe), 1 za LR- zajednički train set sastavljen od trainova svih grupa, testiran na pojedinačnim test setovima svake grupe, 1 SVM pojedinačni train setovi testirani na pojedinačnim test setovima(svake grupe) i još 1 SVM zajednički train set sastavljen od treniranih setova svih grupa te testiran na pojedinačnim test setovima svake grupe. Uz to, Izračunata je točnost, preciznost, odziv i ključna F1-mjera za svaki model.

Plitko duboko učenje (SDL):
* GRM
* LSTM

Korišteni su LSTM modeli za obradu teksta po sekvencama. Nisu korišteni pojedinačni splitovi korpusa, već jedan veliki korpus koji smo splitali direktno u kodu pomoću _StratifiedGroupKFold_. To doprinosi boljoj raspodijeljenosti. Izračunata je točnost, preciznost, odziv i ključna F1-mjera. Sva četiri konačna korpusa su sjedinjena u jedan dokument. Uz pomoć programa osigurano je da  recenzije budu numerirane kontinuirano kroz cijeli dokument

Transformatori:
* BERTić
* Gemma 2

Korišteni su Googleov model Gemma 2 te model BERTić koji je prilagođen hrvatskom jeziku. Kao i za modele dubokog učenja, primijenjena je promjena podatkovnog seta kako bi se osigurala bolja raspodijeljenost. Uz tom, izračunata je točnost, preciznost, odziv i ključna F1-mjera.


Demo
--
 
Uzeli smo modele koji su nam pokazali najbolje rezultate, LR_zajednicki, GRU model i GEMMA spojili trenirane spremljene modele u jednu skrpitu. Napravili smo _Gradio prompt_ gdje se od korisnika traži da upiše rečenicu i onda se prikazuju predviđanja za sva tri modela. Datoteke potrebne za pokretanje demo programa dostupne su u istoimenoj mapi te na [huggingface platformi](https://huggingface.co/spaces/BinaryAnalysts/DEMO-OPJ).

Rezultati
--

Odabrali smo tri najbolje rangirana modela iz svake skupine pristupa: Logistic Regression, GRU i Gemma 2. 
Sljedeća tablica pokazuje njihove konačne rezultate:


|        Model        | Točnost |  Preciznost | Odziv | F1-mjera |
| ------------------- | ------- | ----------- | ----- | -------- |  
| Logistic Regression |   76%   |     72%     |  76%  |   73%    |
| GRU                 |  66,8%  |    67,5%    | 66,8% |  65,9%   |
| Gemma 2             |   85%   |    84,5%    |  85%  |  84,7%   |
 

Najbolje je rezultate ostvario model Gemma 2 -  značajnih 85% točnosti te F1-mjeru od 84,7%. Time je nadmašio klasične modele strojnog učenja i modele dubokog učenja. _Confusion matrix_ pokazuje da svi modeli vrlo uspješno prepoznaju pozitivne i negativne rečenice, što je i očekivano obzirom na to da one čine najveći dio korpusa. Neutralne su rečenice predstavljale veći izazov za sve modele. To može biti zato što dijele vokabular i s pozitivnim i s negativnim recenzijama, zbog čega model može napraviti pogrešku. Najviše se grešaka događa kod klasifikacije sarkazma s obzirom na kompleksnost takvih izraza.

