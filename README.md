Project of Croatian Sentiment Reviews 
--


Članovi ekipe:


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
* Domena: recenzije doktora
* Veličina korpusa: 4070

Za prikupljanje sadržaja korpusa korištena je web-stranica najdoktor.com. Portal omogućava korisnicima komentiranje rada i stručnosti liječnika privatnog i javnog sektora iz različitih medicinskih domena. Komentari služe kao recenzije koje se potom označuju nekom od oznaka, poput pozitivno, negativno i mješovito. Komentari su javno dostupni, a podaci komentatora, poput korisničkog imena, se ne prikupljaju.
Korpus sadržava 4070 rečenica uzetih iz 825 komentara o 25 liječnika.

Pilot anotiranje:
--

Projekt je zahtijevao provođenje pilot-anotacijske kampanje prije konačnog označavanja cijelog skupa. Za potrebe ovog pilota odabran je reprezentativni nasumiačan uzorak od 150 rečenica iz našeg korpusa. Zadatak svakog člana tima bio je neovisno analizirati i označiti emocionalni ton (sentiment) svake rečenice. U našem specifičnom projektu, anotacijska shema proširena je na pet kategorija, čime smo obuhvatili sljedeće tonove: pozitivan, negativan, sarkastičan, mješoviti (mixed) i neutralan. Četiri člana su provela osnovnu anotaciju, dok je peti član proveo završnu anotaciju čime je svakoj rečenici dodijeljena konačna oznaka.

Anotiranje:
--

Treniranje:
--

Metode:
--
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


Alati:
--


