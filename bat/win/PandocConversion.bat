xcopy /s /y C:\Users\Manuel\Dropbox\Apps\Overleaf\Masterarbeit\* D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\tex\
D:
cd D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\tex\
type tex\LELE.bib tex\LELE_bib_manual.bib > tex\References.bib
::powershell -Command "(gc tex\00_Preamble.tex) -replace 'tex/LELE.bib', 'tex/References.bib' | Out-File -encoding ASCII tex\00_Preamble.tex"
::powershell -Command "(gc tex\00_Preamble.tex) -replace '\addbibresource{tex/References.bib_manual.bib}', '' | Out-File -encoding ASCII tex\00_Preamble.tex"
pandoc tex\XX_Main.tex --citeproc -o D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\tex\wrd\Masterarbeit.docx
pause