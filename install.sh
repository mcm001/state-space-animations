sudo apt update

sudo apt-get install -qqy --no-install-recommends apt-utils ffmpeg sox libcairo2-dev texlive texlive-fonts-extra texlive-latex-extra texlive-latex-recommended texlive-science tipa

python3.7 -m pip install setuptools
python3.7 -m pip install -r requirements.txt
python3.7 -m pip install manimlib