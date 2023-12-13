# /sh/shell
gunicorn kfastml.entrypoints.api_server:app -w 4 -k uvicorn.workers.UvicornWorke
