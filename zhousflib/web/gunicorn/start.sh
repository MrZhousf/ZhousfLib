# nohup uvicorn "app.server:app" --host 0.0.0.0 --port 5006 --workers 1 > output.log 2>&1 &
# gunicorn "app.server:app"  -b 0.0.0.0:5007 -w 1 -k uvicorn.workers.UvicornWorker -t 600
gunicorn -c gunicorn_cf.py app.server:app