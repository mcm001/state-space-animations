sudo apt update

sudo apt-get install -qqy --no-install-recommends apt-utils ffmpeg sox libcairo2-dev texlive texlive-fonts-extra texlive-latex-extra texlive-latex-recommended texlive-science tipa

python -m pip install setuptools
python -m pip install -r requirements.txt
python -m pip install manimlib