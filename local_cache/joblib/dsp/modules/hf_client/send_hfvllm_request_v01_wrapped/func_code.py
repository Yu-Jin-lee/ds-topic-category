# first line: 259
@NotebookCacheMemory.cache(ignore=["arg"])
def send_hfvllm_request_v01_wrapped(arg, url, port, **kwargs):
    return send_hftgi_request_v01(arg, url, port, **kwargs)
