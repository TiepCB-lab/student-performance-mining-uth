# Student Performance Mining UTH

Du an khai pha du lieu va du doan ket qua hoc tap sinh vien.

## Cau truc

- `backend/`: API Python FastAPI.
- `frontend/`: giao dien ReactJS, Vite va Tailwind CSS.
- `data/raw/`: du lieu tho ban dau.
- `data/processed/`: du lieu da xu ly.
- `models/`: model da train.
- `notebooks/`: notebook theo pipeline phan tich va huan luyen.
- `training/`: script huan luyen production.
- `reports/`: metrics, chart va bao cao danh gia model.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Huan luyen model

```powershell
python training/train.py
```

## Chay backend

```powershell
cd backend
python run.py
```

## Chay test

```powershell
pytest
```
