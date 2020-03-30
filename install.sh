sudo apt update \
&& apt-get install -qqy --no-install-recommends \
apt-utils \
ffmpeg \
sox \
libcairo2-dev \
texlive \
texlive-fonts-extra \
texlive-latex-extra \
texlive-latex-recommended \
texlive-science \
tipa \
&& pip3 install setuptools
&& pip3 install -r requirements.txt
&& pip3 install manimlib