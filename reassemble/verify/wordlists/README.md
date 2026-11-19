# Word lists for OCR verification

One lowercase word per line, file name = language code (`en.txt`, `yo.txt`, `pcm.txt`, ...).
Select languages with `--lang en,pcm` (any combination; words are merged).

* `en.txt`: derived from the SCOWL-based `wamerican` list (public-domain-style licence),
  lowercase alphabetic entries only.
* `yo.txt` and `pcm.txt` (Yoruba, Nigerian Pidgin) are small **starter lists** of very
  common words, there to show the mechanism. Extend them with a proper corpus before
  relying on them. Yoruba words are listed both with and without tone marks, because
  Tesseract's English model drops diacritics.
