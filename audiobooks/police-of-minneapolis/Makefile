default: police-of-minneapolis.mp3 police-of-minneapolis.opus

minneapolis_findings_report_2023.06.15_0.pdf:
	wget https://www.justice.gov/d9/press-releases/attachments/2023/06/16/minneapolis_findings_report_2023.06.15_0.pdf

text.txt: minneapolis_findings_report_2023.06.15_0.pdf
	pdftoppm -png -singlefile minneapolis_findings_report_2023.06.15_0.pdf cover-art # generates cover-art.png
	pdftohtml -i  -enc UTF-8 minneapolis_findings_report_2023.06.15_0.pdf text.html
	w3m -v -T text/html texts.html > text.txt.orig
	cp text.txt{.orig,}

	false # edit the file manually:
	# drop the dotted lines in table of contents:
	# %s/[.][.]\+.*$//
	# drop http links:
	# %s/http[s]\?.*$//
	# lowercase the all-caps phrases as Mimic spells it by letter:
	# %s/[A-Z ]\{9,\}/\L&/g
	# review of diff and manual cleanup

text.wav: text.txt
	#flite -f text.txt -o text.wav
	mimic3 < text.txt > /tmp/mimic.wav

police-of-minneapolis.mp3: text.wav
	ffmpeg -i text.wav  -i cover-art.png \
	-map 0:0 -map 1:0 -c:a libmp3lame -qscale:a 0 -c:v copy -metadata:s:v title="Album cover" -metadata:s:v comment="Cover (front)" \
	police-of-minneapolis.mp3 -y

police-of-minneapolis.opus: text.wav
	#ffmpeg -i text.wav  police-of-minneapolis.opus # no cover art
	# cover art:
	echo ';FFMETADATA1' > cover-art.ffmetadata
	# https://github.com/jovaska1337/misc/blob/master/mkpblock.py
	./mkpblock.py 3 "" cover-art.png >> cover-art.ffmetadata
	ffmpeg -i text.wav -i cover-art.ffmetadata -map_metadata 1  police-of-minneapolis.opus -y
