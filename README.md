# Real-time video decomposition
> Oleh Hutsuliak, Oleksandr Stadnik, Taras Levytskyi

# Videos
[Taras Levytskyi](https://youtu.be/byIitgK1kJM)  
[Oleh Hutsuliak](https://www.youtube.com/watch?v=Fg1-TmvAm0U)   
[Stadnik Oleksandr](https://www.youtube.com/watch?v=VdXeK0sRoG8)

# Important
For the cuda-accelerated version, python 3.10 must be used.

Use a virtual environment (recommended on Debian/Ubuntu due PEP 668):
```
python3.10 -m venv .venv
source .venv/bin/activate
```
If `python3.10` is not available in your apt repositories (for example Ubuntu 24.04), install Python 3.10 locally via `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/uv" UV_NO_MODIFY_PATH=1 sh
$HOME/.local/uv/uv python install 3.10
$HOME/.local/uv/uv venv --python 3.10 .venv
source .venv/bin/activate
```

Install requirements from requirements.txt:
```
pip install -r requirements.txt
```
Then launch the file:
```
python src/__main__.py
```
