# mata66-tcc


## Setup.: Python 3.10

- Whindows (CMD on project root folder)

$python -m venv venv

$venv\Scripts\activate

$pip install -r requirements.txt

- Linux (terminal on project root folder)

$python3 -m venv venv

$source bin\activate

$pip3 install -r requirements.txt

## Some results preview

<img src="result/movie-lens-small/first_run/ap.png" width="70%" />
<img src="result/movie-lens-small/first_run/map.png" width="70%" />
<img src="result/movie-lens-small/first_run/mrr.png" width="70%" />

<br />
<br />


## Tests

<b>Run all tests</b><br />
$python -m unittest discover

<b>Run only one test class</b><br />
<i>$python -m unittest package.file.ClassName</i><br /><br />
<i>$python -m unittest package.file.ClassName.test_method</i><br /><br />

### References

Cantador, Iván, Peter Brusilovsky, and Tsvi Kuflik. "Second workshop on information heterogeneity and fusion in recommender systems (HetRec2011)." Proceedings of the fifth ACM conference on Recommender systems. 2011. Doi: https://doi.org/10.1145/2043932.2044016