xcopy ..\include src\include /e /i /y
xcopy ..\src src\src /e /i /y
del .\src\Makevars.win
copy .\src\Makevars_fullcode.win .\src\Makevars.win /y
R CMD build --no-build-vignettes .