## CREATE A VIRTUAL ENVIRONMENT
```bash 
!pip install uv
uv init project_name
cd project_name
uv venv
.venv\Scripts\activate
```
## Open main.py in a marimo sandbox
```bash
marimo edit --sandbox main.py
```
## Run the app in a browser
```bash
marimo edit main.py
```

## GITHUB REPOSITORY
```bash
git init
git add README.md
git commit -m "commit message"
git branch -M main
git remote add origin https:<//github.com/username/repository_name.git>
git push -u origin main
```
